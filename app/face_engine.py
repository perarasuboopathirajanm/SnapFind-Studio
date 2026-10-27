# =============================================================================
#  SnapFind Studio - photo delivery by face
#  Copyright (c) 2026 Perarasu Boopathirajan. All rights reserved.
#  Author: Perarasu Boopathirajan
#  Unauthorized copying, redistribution or claiming authorship of this file,
#  via any medium, is prohibited without written permission. See LICENSE.
# =============================================================================
"""Face detection + recognition using OpenCV's YuNet (detector) and SFace (recognizer).

Both models are small ONNX files, run on CPU, and need no GPU or paid API.
"""
import io
import os
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone HEIC support (optional)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
DET_FILE = "face_detection_yunet_2023mar.onnx"
REC_FILE = "face_recognition_sface_2021dec.onnx"
MODEL_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODEL_URLS = {
    DET_FILE: f"{MODEL_BASE}/face_detection_yunet/{DET_FILE}",
    REC_FILE: f"{MODEL_BASE}/face_recognition_sface/{REC_FILE}",
}

# Cosine similarity threshold. OpenCV recommends 0.363 for SFace.
# Raise it (e.g. 0.42) for fewer false matches, lower it for more matches.
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))
DETECT_SCORE = float(os.environ.get("DETECT_SCORE", "0.75"))
MAX_DETECT_SIDE = int(os.environ.get("MAX_DETECT_SIDE", "1920"))
MIN_FACE_PX = int(os.environ.get("MIN_FACE_PX", "28"))


def _ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path) or os.path.getsize(path) < 100_000:
            print(f"[face] downloading {name} ...", flush=True)
            tmp = path + ".part"  # download to a temp name so a failed download never looks complete
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < 100_000:
                os.remove(tmp)
                raise RuntimeError(f"Download of {name} looks broken (too small). Check your internet connection.")
            os.replace(tmp, path)


class FaceEngine:
    def __init__(self):
        _ensure_models()
        self._lock = threading.Lock()  # OpenCV DNN objects are not thread-safe
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(MODEL_DIR, DET_FILE), "", (320, 320), DETECT_SCORE, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(os.path.join(MODEL_DIR, REC_FILE), "")

    @staticmethod
    def load_image(data: bytes) -> np.ndarray:
        """Decode bytes -> BGR array, honouring EXIF rotation (phones!)."""
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _detect(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = min(1.0, MAX_DETECT_SIDE / max(h, w))
        work = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
        self.detector.setInputSize((work.shape[1], work.shape[0]))
        _, faces = self.detector.detect(work)
        return work, ([] if faces is None else list(faces)), scale

    def faces(self, img: np.ndarray):
        """Return list of (embedding[128] float32 normalised, bbox[x,y,w,h] in original px)."""
        with self._lock:
            work, faces, scale = self._detect(img)
            out = []
            for f in faces:
                if min(f[2], f[3]) < MIN_FACE_PX * scale:
                    continue
                aligned = self.recognizer.alignCrop(work, f)
                emb = self.recognizer.feature(aligned).flatten().astype(np.float32)
                emb /= (np.linalg.norm(emb) + 1e-9)
                bbox = [float(v / scale) for v in f[:4]]
                out.append((emb, bbox))
            return out

    def best_face(self, img: np.ndarray):
        """Largest face in the image (used for the selfie)."""
        found = self.faces(img)
        if not found:
            return None
        return max(found, key=lambda x: x[1][2] * x[1][3])


_engine = None
_engine_lock = threading.Lock()


def get_engine() -> FaceEngine:
    # The startup warm-up thread and the first request can both get here;
    # the lock stops two threads downloading/loading the models at once.
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FaceEngine()
    return _engine
