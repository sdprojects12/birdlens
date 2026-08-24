"""
osea_model.py
-------------
OSEA model loader and inference engine.

Uses the OFFICIAL model files from the osea_mobile release page:
  https://github.com/sun-jiao/osea_mobile/releases/tag/assets

Files loaded:
  bird_model.onnx    — quantized ResNet34 bird classifier (DIB-10K, 10,000+ species)
  ssd_mobilenet.onnx — quantized SSD MobileNet bird detector (ONNX Model Zoo)
  bird_info.json     — species label list: [{"latin": ..., "english": ..., ...}, ...]

This module has NO internet calls after the initial model download.
No Flask, no eBird, no iNaturalist, no Wikipedia.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Optional psutil (soft dependency) ────────────────────────────────────────
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── Official model asset URLs (from osea_mobile GitHub releases) ──────────────
ASSET_BASE = "https://github.com/sun-jiao/osea_mobile/releases/download/assets"
ASSETS = {
    "bird_model.onnx":    f"{ASSET_BASE}/bird_model.onnx",
    "ssd_mobilenet.onnx": f"{ASSET_BASE}/ssd_mobilenet.onnx",
    "bird_info.json":     f"{ASSET_BASE}/bird_info.json",
}

# ResNet34 ImageNet preprocessing constants (standard, confirmed by official docs)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CLASSIFIER_INPUT_SIZE = 224   # centre-crop target
_CLASSIFIER_RESIZE     = 256   # resize shorter side to this first

# SSD MobileNet: accepts uint8 [1, H, W, 3], dynamic spatial dims (no fixed size)
# Output node names from the ONNX Model Zoo SSD-MobileNetV1-12 spec:
_SSD_INPUT        = "image_tensor:0"
_SSD_OUT_BOXES    = "detection_boxes:0"
_SSD_OUT_SCORES   = "detection_scores:0"
_SSD_OUT_CLASSES  = "detection_classes:0"
_SSD_OUT_NUM      = "num_detections:0"


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Detection:
    detected: bool
    confidence: float
    box_ymin: float   # normalised [0,1]
    box_xmin: float
    box_ymax: float
    box_xmax: float
    detector_time_s: float

    def box_pixels(self, h: int, w: int):
        """Convert normalised box to pixel coordinates."""
        return (
            int(self.box_ymin * h), int(self.box_xmin * w),
            int(self.box_ymax * h), int(self.box_xmax * w),
        )


@dataclass
class Prediction:
    rank: int
    common_name: str
    scientific_name: str
    raw_score: float           # softmax probability from the classifier
    extra: dict = field(default_factory=dict)   # any other fields from bird_info.json


# ── Utility ───────────────────────────────────────────────────────────────────
def _ram_mb() -> Optional[float]:
    if not _HAS_PSUTIL:
        return None
    return _psutil.Process().memory_info().rss / 1024 / 1024


def download_assets(model_dir: Path, verbose: bool = True) -> None:
    """
    Download official OSEA model assets into model_dir if not already present.
    All downloads are from github.com/sun-jiao/osea_mobile/releases.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ASSETS.items():
        dest = model_dir / filename
        if dest.exists():
            if verbose:
                print(f"  [cached]  {filename}  ({dest.stat().st_size // 1024} KB)")
            continue
        if verbose:
            print(f"  [downloading]  {filename}  from {url} …", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            if verbose:
                print(f"  done  ({dest.stat().st_size // 1024} KB)")
        except Exception as exc:
            if verbose:
                print(f"\n  ERROR: {exc}")
            raise RuntimeError(
                f"Failed to download {filename}.\n"
                f"URL: {url}\n"
                f"You can also download manually and place in: {model_dir}\n"
                f"Original error: {exc}"
            ) from exc


# ── Main class ────────────────────────────────────────────────────────────────
class OSEAModel:
    """
    Loads and runs OSEA inference.

    Usage:
        model = OSEAModel(model_dir=Path("models"))
        model.load()                          # download + initialise; call once
        det, preds = model.predict(img_path)  # call as many times as you like
    """

    def __init__(self, model_dir: Path = Path("models"), detection_threshold: float = 0.5):
        self.model_dir = Path(model_dir)
        self.detection_threshold = detection_threshold

        self._classifier_session = None
        self._detector_session   = None
        self._labels: list[Any] = []

        self.num_species: int = 0
        self.classifier_load_s: float = 0.0
        self.detector_load_s:   float = 0.0
        self.total_load_s:      float = 0.0
        self.ram_before_mb: Optional[float] = None
        self.ram_after_mb:  Optional[float] = None

    # ── Loading ────────────────────────────────────────────────────────────
    def load(self, verbose: bool = True) -> None:
        """Download (if needed) and initialise all models."""
        import onnxruntime as ort

        self.ram_before_mb = _ram_mb()
        t_total = time.perf_counter()

        # 1. Download assets
        if verbose:
            print("\nDownloading / verifying model assets …")
        download_assets(self.model_dir, verbose=verbose)

        # 2. Load labels
        with open(self.model_dir / "bird_info.json", encoding="utf-8") as f:
            self._labels = json.load(f)
        self.num_species = len(self._labels)

        # 3. Load SSD MobileNet detector
        if verbose:
            print(f"\nLoading detector  (ssd_mobilenet.onnx) …", end="", flush=True)
        t0 = time.perf_counter()
        self._detector_session = ort.InferenceSession(
            str(self.model_dir / "ssd_mobilenet.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.detector_load_s = time.perf_counter() - t0
        if verbose:
            print(f"  {self.detector_load_s:.2f}s")

        # 4. Load ResNet34 bird classifier
        if verbose:
            print(f"Loading classifier (bird_model.onnx) …", end="", flush=True)
        t0 = time.perf_counter()
        self._classifier_session = ort.InferenceSession(
            str(self.model_dir / "bird_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.classifier_load_s = time.perf_counter() - t0
        if verbose:
            print(f"  {self.classifier_load_s:.2f}s")

        self.total_load_s  = time.perf_counter() - t_total
        self.ram_after_mb  = _ram_mb()

    # ── Preprocessing ───────────────────────────────────────────────────────
    def _preprocess_classify(self, img_array: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Official ResNet34 preprocessing (from torchvision docs and OSEA paper):
          1. Resize shorter edge to 256, bilinear
          2. Centre-crop to 224×224
          3. Convert to float32, normalise to [0,1]
          4. Apply ImageNet mean/std normalisation
          5. Transpose to NCHW

        Returns (tensor [1,3,224,224], preprocessing_time_s).
        """
        from PIL import Image

        t0 = time.perf_counter()
        if isinstance(img_array, np.ndarray):
            img = Image.fromarray(img_array.astype(np.uint8))
        else:
            img = img_array   # already PIL

        # Resize shorter side to 256
        w, h = img.size
        if h < w:
            new_h, new_w = _CLASSIFIER_RESIZE, int(w * _CLASSIFIER_RESIZE / h)
        else:
            new_h, new_w = int(h * _CLASSIFIER_RESIZE / w), _CLASSIFIER_RESIZE
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Centre crop 224×224
        w, h  = img.size
        left  = (w - _CLASSIFIER_INPUT_SIZE) // 2
        top   = (h - _CLASSIFIER_INPUT_SIZE) // 2
        img   = img.crop((left, top, left + _CLASSIFIER_INPUT_SIZE, top + _CLASSIFIER_INPUT_SIZE))

        arr = np.array(img, dtype=np.float32) / 255.0   # [224,224,3]  [0,1]
        arr = (arr - _MEAN) / _STD                       # normalise
        arr = arr.transpose(2, 0, 1)[np.newaxis]         # [1,3,224,224]

        return arr, time.perf_counter() - t0

    def _preprocess_detect(self, img_array: np.ndarray) -> np.ndarray:
        """
        SSD MobileNet preprocessing: uint8 [1, H, W, 3] — no resizing needed,
        the ONNX model accepts dynamic spatial dimensions.
        """
        return img_array[np.newaxis].astype(np.uint8)

    # ── Detector ────────────────────────────────────────────────────────────
    def _run_detector(self, img_array: np.ndarray) -> Detection:
        """
        Run SSD MobileNet on the full image.
        Returns the highest-confidence detection (or a 'not detected' result).

        SSD-MobileNetV1-12 ONNX output names (from ONNX Model Zoo):
          num_detections:0   — float [1, 1]
          detection_boxes:0  — float [1, 100, 4]  (ymin, xmin, ymax, xmax normalised)
          detection_scores:0 — float [1, 100]
          detection_classes:0— float [1, 100]
        """
        t0 = time.perf_counter()

        inp = self._preprocess_detect(img_array)

        # Discover actual input name from the session (defensive)
        input_name = self._detector_session.get_inputs()[0].name

        outputs = self._detector_session.run(None, {input_name: inp})

        # Map outputs by name for robustness
        out_names = [o.name for o in self._detector_session.get_outputs()]
        out_map = dict(zip(out_names, outputs))

        def _get(candidates):
            for c in candidates:
                if c in out_map:
                    return out_map[c]
            # fallback: use positional order if names differ
            return None

        scores  = _get([_SSD_OUT_SCORES,  "detection_scores:0",  "detection_scores"])
        boxes   = _get([_SSD_OUT_BOXES,   "detection_boxes:0",   "detection_boxes"])

        det_time = time.perf_counter() - t0

        if scores is None or boxes is None:
            return Detection(False, 0.0, 0, 0, 0, 0, det_time)

        scores = np.array(scores).flatten()
        boxes  = np.array(boxes).reshape(-1, 4)

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        box        = boxes[best_idx]   # [ymin, xmin, ymax, xmax] normalised

        detected = best_score >= self.detection_threshold
        return Detection(
            detected     = detected,
            confidence   = best_score,
            box_ymin     = float(box[0]),
            box_xmin     = float(box[1]),
            box_ymax     = float(box[2]),
            box_xmax     = float(box[3]),
            detector_time_s = det_time,
        )

    # ── Classifier ──────────────────────────────────────────────────────────
    def _run_classifier(self, region: np.ndarray, k: int = 5) -> tuple[list[Prediction], float, float]:
        """
        Run ResNet34 classifier on a cropped (or full) image region.

        Returns (predictions, preprocess_s, classifier_s).
        """
        tensor, preprocess_s = self._preprocess_classify(region)

        input_name = self._classifier_session.get_inputs()[0].name

        t0 = time.perf_counter()
        logits = self._classifier_session.run(None, {input_name: tensor})[0]  # [1, num_species]
        classifier_s = time.perf_counter() - t0

        # Softmax
        logits = logits.flatten().astype(np.float64)
        logits -= logits.max()           # numerical stability
        exp    = np.exp(logits)
        probs  = exp / exp.sum()

        top_k_idx = probs.argsort()[-k:][::-1]

        predictions = []
        for rank, idx in enumerate(top_k_idx, start=1):
            common_name, scientific_name, extra = self._parse_label(
                self._labels[idx] if idx < len(self._labels) else None,
                idx,
            )
            predictions.append(Prediction(
                rank           = rank,
                common_name    = common_name,
                scientific_name= scientific_name,
                raw_score      = float(probs[idx]),
                extra          = extra,
            ))

        return predictions, preprocess_s, classifier_s

    def _parse_label(self, label: Any, idx: int) -> tuple[str, str, dict]:
        """
        Convert one bird_info.json entry into Prediction fields.

        Supported label shapes:
          - dict: {"english": ..., "latin": ...}
          - list/tuple: [local_name, common_name, scientific_name, ...]
          - str: common_name
        """
        if isinstance(label, dict):
            common_name = label.get("english", label.get("name", f"Species {idx}"))
            scientific_name = label.get("latin", label.get("scientific", ""))
            extra = {
                k: v for k, v in label.items()
                if k not in ("english", "name", "latin", "scientific")
            }
            return str(common_name), str(scientific_name), extra

        if isinstance(label, (list, tuple)):
            common_name = label[1] if len(label) > 1 else (label[0] if label else f"Species {idx}")
            scientific_name = label[2] if len(label) > 2 else ""
            extra = {}
            if len(label) > 0:
                extra["local_name"] = label[0]
            if len(label) > 3:
                extra["other"] = list(label[3:])
            return str(common_name), str(scientific_name), extra

        if isinstance(label, str):
            return label, "", {}

        return f"Species {idx}", "", {}

    # ── Public predict ──────────────────────────────────────────────────────
    def predict(
        self,
        image_path: str | Path,
        k: int = 5,
        use_detector: bool = True,
    ) -> tuple[Detection, list[Prediction], dict]:
        """
        Full pipeline: detect → crop → classify.

        Returns
        -------
        detection   : Detection
        predictions : list[Prediction]   top-k, sorted best-first
        timing      : dict with keys detector_s, preprocess_s, classifier_s, total_s
        """
        if self._classifier_session is None:
            raise RuntimeError("Call .load() before .predict()")

        from PIL import Image

        img_path  = Path(image_path)
        pil_image = Image.open(img_path).convert("RGB")
        img_array = np.array(pil_image)
        h, w = img_array.shape[:2]

        t_total = time.perf_counter()

        # ── Detection ──
        if use_detector and self._detector_session is not None:
            detection = self._run_detector(img_array)
        else:
            detection = Detection(False, 0.0, 0.0, 0.0, 1.0, 1.0,
                                  detector_time_s=0.0)

        # ── Crop if bird detected ──
        if detection.detected:
            y0, x0, y1, x1 = detection.box_pixels(h, w)
            # add 5% padding, clamp to image bounds
            pad_y = max(1, int((y1 - y0) * 0.05))
            pad_x = max(1, int((x1 - x0) * 0.05))
            y0 = max(0, y0 - pad_y)
            x0 = max(0, x0 - pad_x)
            y1 = min(h, y1 + pad_y)
            x1 = min(w, x1 + pad_x)
            region = img_array[y0:y1, x0:x1]
        else:
            region = img_array   # classify full image if no bird found

        # ── Classify ──
        predictions, preprocess_s, classifier_s = self._run_classifier(region, k=k)

        total_s = time.perf_counter() - t_total

        timing = {
            "detector_s":    detection.detector_time_s,
            "preprocess_s":  preprocess_s,
            "classifier_s":  classifier_s,
            "total_s":       total_s,
        }

        return detection, predictions, timing