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

# ── Memory diagnostics (lightweight) ──────────────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def _log_memory(label: str) -> None:
    """Log current process memory usage."""
    if _HAS_PSUTIL:
        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        log.info(f"MEMORY [{label}] {rss_mb:.1f} MB")

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")

# ── Config ────────────────────────────────────────────────────────────────────
# Load API key from environment first; fall back to hardcoded value.
# Preferred: set EBIRD_API_KEY in your shell so the key is never in source.
EBIRD_API_KEY = os.environ.get("EBIRD_API_KEY", "")
EBIRD_BASE = "https://api.ebird.org/v2"

# Upload limits
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB hard cap
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

# Whitelist of valid eBird region codes (2-letter country or country-state)
REGION_RE = re.compile(r"^[A-Z]{2}(-[A-Z0-9]{1,3})?$")

# External request timeout
TIMEOUT = 12

# OSEA model files live here (downloaded automatically on first load)
OSEA_MODEL_DIR = PROJECT_ROOT / "models"

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


# ── OSEA ──────────────────────────────────────────────────────────────────────
# Loaded once at application startup and reused for every request (see bottom
# of this file). Do NOT construct/load a new OSEAModel per-request.
from osea_model import OSEAModel
from osea_confidence import decide, IdentificationState, DEFAULT_THRESHOLDS

OSEA_MODEL: OSEAModel | None = None


def get_osea_model() -> OSEAModel:
    """
    Return the process-wide OSEA model instance. Loaded once via
    load_osea_model() at startup; this getter just guards against the
    (unexpected) case of a route firing before startup finished.
    """
    global OSEA_MODEL
    if OSEA_MODEL is None:
        log.warning("OSEA model accessed before startup load completed — loading now")
        OSEA_MODEL = OSEAModel(model_dir=OSEA_MODEL_DIR)
        OSEA_MODEL.load(verbose=True)
    return OSEA_MODEL


def load_osea_model() -> None:
    """Load SSD MobileNet detector + OSEA classifier + bird_info.json once."""
    global OSEA_MODEL
    _log_memory("before OSEA load")
    log.info("Loading OSEA models (detector + classifier + labels)")
    model = OSEAModel(model_dir=OSEA_MODEL_DIR)
    model.load(verbose=True)
    OSEA_MODEL = model
    _log_memory("after OSEA load")
    log.info(
        "OSEA ready: %d species loaded from bird_info.json", model.num_species
    )


def classify_with_osea(image_path: str, top_k: int = 5) -> dict:
    """
    Run the OSEA pipeline (detect -> crop -> classify) on one image and
    normalize the result into the same predictions-list shape BioCLIP used
    to produce, plus an explicit confidence decision.

    Returns a dict:
      {
        "state": "not_a_bird" | "unable_to_identify" | "identified",
        "reason": str,
        "detector_confidence": float,
        "top1_score": float,
        "top2_score": float,
        "margin": float,
        "predictions": [ {common_name, scientific_name, score, ...}, ... ],
      }
    """
    try:
        _log_memory("before prediction")
        model = get_osea_model()
        detection, predictions, timing = model.predict(image_path, k=top_k, use_detector=True)

        results = []
        for p in predictions:
            results.append({
                "common_name": sanitize_text(p.common_name, 120) or "Unknown",
                "scientific_name": sanitize_text(p.scientific_name, 120),
                # Keep the 0-100 scale the existing frontend/API already expects.
                "score": round(float(p.raw_score) * 100, 2),
            })

        top1 = predictions[0].raw_score if predictions else 0.0
        top2 = predictions[1].raw_score if len(predictions) > 1 else 0.0

        decision = decide(
            detector_confidence=detection.confidence,
            detector_detected=detection.detected,
            top1_score=top1,
            top2_score=top2,
            thresholds=DEFAULT_THRESHOLDS,
        )

        _log_memory("after prediction/API processing")
        return {
            "state": decision.state.value,
            "reason": decision.reason,
            "detector_detected": detection.detected,
            "detector_confidence": round(float(decision.detector_confidence), 4),
            "detector_box": {
                "ymin": round(float(detection.box_ymin), 4),
                "xmin": round(float(detection.box_xmin), 4),
                "ymax": round(float(detection.box_ymax), 4),
                "xmax": round(float(detection.box_xmax), 4),
            },
            "top1_score": round(float(decision.top1_score), 4),
            "top2_score": round(float(decision.top2_score), 4),
            "margin": round(float(decision.margin), 4),
            "predictions": results,
            "timing_ms": {
                "detector": round(timing["detector_s"] * 1000, 1),
                "preprocess": round(timing["preprocess_s"] * 1000, 1),
                "classifier": round(timing["classifier_s"] * 1000, 1),
                "total": round(timing["total_s"] * 1000, 1),
            },
        }
    except Exception as e:
        log.exception("OSEA error")
        return {
            "state": IdentificationState.UNABLE_TO_IDENTIFY.value,
            "reason": f"OSEA failed: {e}",
            "detector_detected": False,
            "detector_confidence": 0.0,
            "detector_box": {"ymin": 0.0, "xmin": 0.0, "ymax": 0.0, "xmax": 0.0},
            "top1_score": 0.0,
            "top2_score": 0.0,
            "margin": 0.0,
            "predictions": [],
            "error": f"OSEA failed: {e}",
        }


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
                "common_name": sanitize_text(t.get("comName", ""), 120),
                "scientific_name": sanitize_text(t.get("sciName", ""), 120),
                "species_code": code,
                "order": sanitize_text(t.get("order", ""), 80),
                "family_name": sanitize_text(t.get("familyComName", ""), 80),
                "family_code": sanitize_text(t.get("familyCode", ""), 20),
                "category": sanitize_text(t.get("category", ""), 40),
                "taxon_order": t.get("taxonOrder", ""),
                "ebird_url": f"https://ebird.org/species/{code}" if code else "",
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
                "date": sanitize_text(s.get("obsDt", ""), 30),
                "count": s.get("howMany", "?"),
                "lat": s.get("lat"),
                "lng": s.get("lng"),
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
                "extract": sanitize_text(d.get("extract", ""), 700),
                "image_url": d.get("thumbnail", {}).get("source", ""),
                "wiki_url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
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
                "observations_count": int(t.get("observations_count", 0) or 0),
                "wikipedia_url": t.get("wikipedia_url", ""),
                "photo": t.get("default_photo", {}).get("medium_url", ""),
            }
    except Exception as e:
        log.warning("iNat error: %s", e)
    return {}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
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
    _log_memory("identify: start")
    # ── File presence ──
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    image_filename = img_file.filename or "<unnamed>"

    # ── Size check (read once into memory first) ──
    img_file.stream.seek(0, 2)  # seek to end
    size = img_file.stream.tell()
    img_file.stream.seek(0)     # rewind
    if size > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File too large (max 20 MB)"}), 413

    # ── MIME type check ──
    mime = img_file.content_type or ""
    if mime not in ALLOWED_MIMETYPES:
        return jsonify({"error": "Unsupported file type"}), 415

    # ── Region validation ──
    region = validate_region(request.form.get("region", "IN-MH"))

    # ── Save to temp file with safe extension ──
    ext = safe_ext(img_file.filename)
    tmp_name = f"osea_{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)

    try:
        img_file.save(tmp_path)
        _log_memory("identify: after save")
        result = classify_with_osea(tmp_path, top_k=5)
        _log_memory("identify: after classify")
        prediction_log = ", ".join(
            f"{p['common_name']}={p['score']:.2f}%" for p in result["predictions"]
        ) or "none"
        box = result["detector_box"]
        timing = result.get("timing_ms", {})
        log.info(
            "\nOSEA /identify | file=%s\n"
            "  detector: bird=%s confidence=%.4f box=(ymin=%.4f, xmin=%.4f, ymax=%.4f, xmax=%.4f)\n"
            "  classifier top5: %s\n"
            "  margin: top1-top2=%.4f\n"
            "  decision: state=%s reason=%s\n"
            "  timing_ms: detector=%.1f classifier=%.1f total=%.1f",
            image_filename,
            "YES" if result["detector_detected"] else "NO",
            result["detector_confidence"],
            box["ymin"], box["xmin"], box["ymax"], box["xmax"],
            prediction_log,
            result["margin"],
            result["state"],
            result["reason"],
            timing.get("detector", 0.0),
            timing.get("classifier", 0.0),
            timing.get("total", 0.0),
        )
    except Exception as e:
        log.error("identify error: %s", e)
        return jsonify({"error": "Server error during classification"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        return jsonify({"error": result["error"]}), 500

    state = result["state"]
    predictions = result["predictions"]

    # Not a bird / not confident enough -> tell the frontend plainly and
    # skip the eBird/Wikipedia/iNat lookups (nothing to look up yet).
    if state == IdentificationState.NOT_A_BIRD.value:
        return jsonify({
            "state": state,
            "message": "We couldn't detect a bird in this image.",
            "confidence": {
                "detector_confidence": result["detector_confidence"],
                "top1_score": result["top1_score"],
                "top2_score": result["top2_score"],
                "margin": result["margin"],
                "reason": result["reason"],
            },
            "predictions": predictions,
            "region": region,
        })

    if state == IdentificationState.UNABLE_TO_IDENTIFY.value:
        return jsonify({
            "state": state,
            "message": "A bird appears to be present, but we couldn't determine its species.",
            "confidence": {
                "detector_confidence": result["detector_confidence"],
                "top1_score": result["top1_score"],
                "top2_score": result["top2_score"],
                "margin": result["margin"],
                "reason": result["reason"],
            },
            "predictions": predictions,  # still returned for reference/debugging
            "region": region,
        })

    # ── Identified: preserve existing enrichment pipeline ──
    if not predictions:
        return jsonify({"error": "Classification failed"}), 500

    best = predictions[0]
    ebird = get_ebird_taxonomy(best["scientific_name"])

    sightings = []
    if ebird and ebird.get("species_code"):
        sightings = get_recent_sightings(ebird["species_code"], region)

    wiki = get_wikipedia_summary(best["common_name"])
    inat = get_inat_info(best["scientific_name"])

    return jsonify({
        "state": state,
        "predictions": predictions,
        "confidence": {
            "detector_confidence": result["detector_confidence"],
            "top1_score": result["top1_score"],
            "top2_score": result["top2_score"],
            "margin": result["margin"],
            "reason": result["reason"],
        },
        "ebird": ebird,
        "sightings": sightings,
        "wikipedia": wiki,
        "inat": inat,
        "region": region,
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
                "common_name": sanitize_text(s.get("comName", ""), 120),
                "scientific_name": sanitize_text(s.get("sciName", ""), 120),
                "location": sanitize_text(s.get("locName", ""), 200),
                "date": sanitize_text(s.get("obsDt", ""), 30),
                "count": s.get("howMany", "?"),
                "species_code": sanitize_text(s.get("speciesCode", ""), 20),
            }
            for s in r.json()[:12]
            if isinstance(s, dict)
        ]
        return jsonify({"birds": birds, "region": region})
    except Exception as e:
        log.warning("nearby error: %s", e)
        return jsonify({"error": "Could not fetch sightings"}), 500


if __name__ == "__main__":
    # Load OSEA once, at process startup, before serving any requests.
    load_osea_model()

    # Never run debug=True in production
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
else:
    # Loaded under a WSGI server (gunicorn, per the Procfile) -- gunicorn
    # imports this module once per worker process, so this still satisfies
    # "load once, reuse across requests" per worker.
    load_osea_model()