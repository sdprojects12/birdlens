import os
import re
import sys
import uuid
import tempfile
import logging
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")

# ── Config ────────────────────────────────────────────────────────────────────
# Load API key from environment first; fall back to hardcoded value.
# Preferred: set EBIRD_API_KEY in your shell so the key is never in source.
EBIRD_API_KEY = os.environ.get("EBIRD_API_KEY", "k0rnvug008l9")
EBIRD_BASE    = "https://api.ebird.org/v2"

# Upload limits
MAX_UPLOAD_BYTES  = 20 * 1024 * 1024          # 20 MB hard cap
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
ALLOWED_EXTENSIONS= {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

# Whitelist of valid eBird region codes (2-letter country or country-state)
REGION_RE = re.compile(r"^[A-Z]{2}(-[A-Z0-9]{1,3})?$")

# External request timeout
TIMEOUT = 12

# ── Helpers ───────────────────────────────────────────────────────────────────

def ebird_headers() -> dict:
    return {"X-eBirdApiToken": EBIRD_API_KEY}


def validate_region(region: str) -> str:
    """Return region if valid, else default."""
    r = (region or "").strip().upper()
    return r if REGION_RE.match(r) else "IN-MH"


def safe_ext(filename: str) -> str:
    """Return a safe, allowlisted file extension."""
    ext = Path(secure_filename(filename or "")).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".jpg"


def sanitize_text(value, max_len: int = 512) -> str:
    """Strip dangerous characters and truncate."""
    if not isinstance(value, str):
        return ""
    return value[:max_len].replace("<", "").replace(">", "").replace("&", "and")


# ── BioCLIP ───────────────────────────────────────────────────────────────────

BIOCLIP_CLASSIFIER = None


def get_bioclip_classifier():
    global BIOCLIP_CLASSIFIER
    if BIOCLIP_CLASSIFIER is None:
        from bioclip import Rank, TreeOfLifeClassifier

        log.info("Loading BioCLIP classifier")
        classifier = TreeOfLifeClassifier()
        bird_filter = classifier.create_taxa_filter(Rank.CLASS, ["Aves"])
        classifier.apply_filter(bird_filter)
        log.info("BioCLIP classifier loaded with Aves-only label filter")
        BIOCLIP_CLASSIFIER = classifier
    return BIOCLIP_CLASSIFIER


def classify_with_bioclip(image_path: str, top_k: int = 5) -> list:
    try:
        from bioclip import Rank
        classifier = get_bioclip_classifier()
        log.info("Running BioCLIP on %s", image_path)
        predictions = classifier.predict(image_path, Rank.SPECIES, k=top_k)
        results = []
        for pred in predictions:
            genus   = sanitize_text(pred.get("genus",   ""), 80)
            species = sanitize_text(pred.get("species_epithet", ""), 80)
            scientific_name = sanitize_text(pred.get("species", ""), 120)
            if not scientific_name:
                scientific_name = f"{genus} {species}".strip()
            results.append({
                "common_name":     sanitize_text(pred.get("common_name", "Unknown"), 120),
                "scientific_name": scientific_name,
                "genus":           genus,
                "species":         species,
                "family":          sanitize_text(pred.get("family", ""), 80),
                "order":           sanitize_text(pred.get("order",  ""), 80),
                "score":           round(float(pred.get("score", 0)) * 100, 2),
            })
        return results
    except ImportError:
        log.exception("pybioclip is not installed in the Python environment running Flask")
        return [{"error": "BioCLIP is not installed in the Python environment running Flask. Activate the project venv and install requirements."}]
    except Exception as e:
        log.exception("BioCLIP error")
        return [{"error": f"BioCLIP failed: {e}"}]


# ── eBird ─────────────────────────────────────────────────────────────────────

def get_ebird_taxonomy(scientific_name: str) -> dict:
    if not scientific_name:
        return {}
    try:
        r = requests.get(
            f"{EBIRD_BASE}/ref/taxonomy/ebird",
            params={"fmt": "json", "species": scientific_name, "locale": "en"},
            headers=ebird_headers(), timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list):
            t = data[0]
            code = sanitize_text(t.get("speciesCode", ""), 20)
            return {
                "common_name":     sanitize_text(t.get("comName", ""),      120),
                "scientific_name": sanitize_text(t.get("sciName", ""),      120),
                "species_code":    code,
                "order":           sanitize_text(t.get("order", ""),         80),
                "family_name":     sanitize_text(t.get("familyComName", ""), 80),
                "family_code":     sanitize_text(t.get("familyCode", ""),    20),
                "category":        sanitize_text(t.get("category", ""),      40),
                "taxon_order":     t.get("taxonOrder", ""),
                "ebird_url":       f"https://ebird.org/species/{code}" if code else "",
            }
    except Exception as e:
        log.warning("eBird taxonomy error: %s", e)
    return {}


def get_recent_sightings(species_code: str, region: str = "IN-MH") -> list:
    if not species_code or not REGION_RE.match(region):
        return []
    try:
        r = requests.get(
            f"{EBIRD_BASE}/data/obs/{region}/recent/{species_code}",
            params={"maxResults": 8, "includeProvisional": True},
            headers=ebird_headers(), timeout=TIMEOUT,
        )
        r.raise_for_status()
        return [
            {
                "location": sanitize_text(s.get("locName", ""), 200),
                "date":     sanitize_text(s.get("obsDt",   ""),  30),
                "count":    s.get("howMany", "?"),
                "lat":      s.get("lat"),
                "lng":      s.get("lng"),
            }
            for s in r.json()[:8]
            if isinstance(s, dict)
        ]
    except Exception as e:
        log.warning("eBird sightings error: %s", e)
    return []


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def get_wikipedia_summary(common_name: str) -> dict:
    if not common_name:
        return {}
    safe_name = re.sub(r"[^a-zA-Z0-9 _\-]", "", common_name).replace(" ", "_")[:80]
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_name}",
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json()
            return {
                "extract":   sanitize_text(d.get("extract", ""), 700),
                "image_url": d.get("thumbnail", {}).get("source", ""),
                "wiki_url":  d.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
    except Exception as e:
        log.warning("Wikipedia error: %s", e)
    return {}


# ── iNaturalist ───────────────────────────────────────────────────────────────

def get_inat_info(scientific_name: str) -> dict:
    if not scientific_name:
        return {}
    safe_q = re.sub(r"[^a-zA-Z0-9 ]", "", scientific_name)[:80]
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": safe_q, "rank": "species"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and isinstance(results, list):
            t = results[0]
            cs = t.get("conservation_status") or {}
            return {
                "conservation_status": sanitize_text(cs.get("status_name", ""), 60),
                "observations_count":  int(t.get("observations_count", 0) or 0),
                "wikipedia_url":       t.get("wikipedia_url", ""),
                "photo":               t.get("default_photo", {}).get("medium_url", ""),
            }
    except Exception as e:
        log.warning("iNat error: %s", e)
    return {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: blob: https:; "
        "connect-src 'self';"
    )
    return response


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/identify", methods=["POST"])
def identify():
    # ── File presence ──
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]

    # ── Size check (read once into memory first) ──
    img_file.stream.seek(0, 2)          # seek to end
    size = img_file.stream.tell()
    img_file.stream.seek(0)             # rewind
    if size > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File too large (max 20 MB)"}), 413

    # ── MIME type check ──
    mime = img_file.content_type or ""
    if mime not in ALLOWED_MIMETYPES:
        return jsonify({"error": "Unsupported file type"}), 415

    # ── Region validation ──
    region = validate_region(request.form.get("region", "IN-MH"))

    # ── Save to temp file with safe extension ──
    ext      = safe_ext(img_file.filename)
    tmp_name = f"bioclip_{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)

    try:
        img_file.save(tmp_path)
        predictions = classify_with_bioclip(tmp_path, top_k=5)
    except Exception as e:
        log.error("identify error: %s", e)
        return jsonify({"error": "Server error during classification"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not predictions or "error" in predictions[0]:
        return jsonify({"error": predictions[0].get("error", "Classification failed")}), 500

    best = predictions[0]

    ebird    = get_ebird_taxonomy(best["scientific_name"])
    sightings = []
    if ebird and ebird.get("species_code"):
        sightings = get_recent_sightings(ebird["species_code"], region)

    wiki = get_wikipedia_summary(best["common_name"])
    inat = get_inat_info(best["scientific_name"])

    return jsonify({
        "predictions": predictions,
        "ebird":       ebird,
        "sightings":   sightings,
        "wikipedia":   wiki,
        "inat":        inat,
        "region":      region,
    })


@app.route("/nearby", methods=["GET"])
def nearby():
    region = validate_region(request.args.get("region", "IN-MH"))
    try:
        r = requests.get(
            f"{EBIRD_BASE}/data/obs/{region}/recent/notable",
            params={"maxResults": 12, "detail": "full"},
            headers=ebird_headers(), timeout=TIMEOUT,
        )
        r.raise_for_status()
        birds = [
            {
                "common_name":     sanitize_text(s.get("comName",     ""), 120),
                "scientific_name": sanitize_text(s.get("sciName",     ""), 120),
                "location":        sanitize_text(s.get("locName",     ""), 200),
                "date":            sanitize_text(s.get("obsDt",       ""),  30),
                "count":           s.get("howMany", "?"),
                "species_code":    sanitize_text(s.get("speciesCode", ""),  20),
            }
            for s in r.json()[:12]
            if isinstance(s, dict)
        ]
        return jsonify({"birds": birds, "region": region})
    except Exception as e:
        log.warning("nearby error: %s", e)
        return jsonify({"error": "Could not fetch sightings"}), 500


if __name__ == "__main__":
    # Never run debug=True in production
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
