"""
Local API bridge for the Sijil HTML application.

It keeps Mustakhrij-OCR unchanged and exposes:
    POST /api/verify-document
    GET  /api/health

The response contains OCR output, template-match scores, an explainable
integration decision, and base64 PNGs for the Sijil visualization panel.

Run:
    pip install flask flask-cors opencv-python numpy
    python sijil_verification_backend.py
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import cv2
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from document_integration import (
    AuthenticityScorer,
    TemplateGallery,
    integration_decision,
    read_bgr,
    run_existing_ocr,
)


PROJECT_DIR = Path(__file__).resolve().parent
OCR_DIR = Path(
    os.environ.get("SIJIL_OCR_DIR", str(PROJECT_DIR / "Mustakhrij-OCR"))
).expanduser().resolve()
TEMPLATES_DIR = Path(
    os.environ.get(
        "SIJIL_TEMPLATES_DIR",
        str(PROJECT_DIR / "fakepassports" / "templates"),
    )
).expanduser().resolve()
AUTHENTICITY_MODEL_PATH = Path(
    os.environ.get(
        "SIJIL_AUTHENTICITY_MODEL",
        str(PROJECT_DIR / "fakepassports" / "authenticity_clf.joblib"),
    )
).expanduser().resolve()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_BROWSER_ORIGINS = {
    "null",  # file:// pages use an opaque origin in modern browsers.
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

gallery = TemplateGallery(TEMPLATES_DIR)
authenticity_scorer = AuthenticityScorer(AUTHENTICITY_MODEL_PATH)
if not authenticity_scorer.available:
    print(f"[WARN] Authenticity scoring unavailable: {authenticity_scorer.load_error}")
    print("[WARN] The OCR and template-match flow will still work normally.")


@app.after_request
def add_cors_headers(response):
    # Only allow the local UI (or a file:// page), not arbitrary websites.
    origin = request.headers.get("Origin")
    if origin in ALLOWED_BROWSER_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"
    return response


def png_data_url(image_bgr_or_rgb, rgb=False):
    if image_bgr_or_rgb is None:
        return None
    image = image_bgr_or_rgb
    height, width = image.shape[:2]
    preview_scale = min(1.0, 1600.0 / max(height, width))
    if preview_scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA,
        )
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    encoded64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded64}"


def public_match(result):
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_") and key not in {"aligned_bgr"}
    }


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "ocr_dir": str(OCR_DIR.resolve()),
            "templates_dir": str(TEMPLATES_DIR.resolve()),
            "template_count": len(gallery.templates),
            "matcher": gallery.matcher_name,
            "authenticity_model_loaded": authenticity_scorer.available,
            "authenticity_model_error": authenticity_scorer.load_error,
        }
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(
        {
            "ok": False,
            "error": "Upload is too large. Choose an image smaller than 12 MB.",
        }
    ), 413


@app.route("/api/verify-document", methods=["POST", "OPTIONS"])
def verify_document():
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "No image file was uploaded."}), 400

    suffix = Path(uploaded.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"ok": False, "error": "Unsupported image format."}), 400

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Close the handle before OpenCV/Ollama reopen the path. This is
        # required on Windows, where an open NamedTemporaryFile can be locked.
        uploaded.save(str(temp_path))
        # Read dimensions from the header first so a tiny compressed image
        # cannot force OpenCV to allocate an enormous decoded image.
        try:
            with Image.open(temp_path) as image_header:
                width, height = image_header.size
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ):
            return jsonify(
                {
                    "ok": False,
                    "error": "The uploaded file is not a readable image.",
                }
            ), 400
        if width * height > MAX_IMAGE_PIXELS:
            return jsonify(
                {
                    "ok": False,
                    "error": "Image dimensions are too large. Resize it and try again.",
                }
            ), 413

        # Reject unreadable images before invoking Ollama.
        image = cv2.imread(str(temp_path), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify(
                {
                    "ok": False,
                    "error": "The uploaded file is not a readable image.",
                }
            ), 400
        # OCR failure should be visible but should not prevent a visual
        # template check from running.
        try:
            ocr_result = run_existing_ocr(temp_path, OCR_DIR)
        except Exception as exc:
            ocr_result = {
                "status": "FLAG",
                "reason": f"OCR unavailable: {type(exc).__name__}: {exc}",
                "record": {},
            }

        record = ocr_result.get("record", {})
        try:
            match_result = gallery.match(temp_path, record)
        except (cv2.error, ValueError) as exc:
            return jsonify(
                {
                    "ok": False,
                    "error": f"Image processing failed: {exc}",
                }
            ), 400
        match_public = public_match(match_result)

        # Authenticity scoring runs on the original (unaligned) image, matching
        # how the classifier was trained/used in fakepassports/run_locally.py.
        # A failure here (e.g. model file missing) must not block the
        # OCR/template result the rest of the page already depends on.
        try:
            authenticity_result = authenticity_scorer.score(image)
        except Exception as exc:
            authenticity_result = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

        decision = integration_decision(ocr_result, match_public, authenticity_result)
        match_visual, template_preview = gallery.visualization(match_result)

        response = {
            "ok": True,
            "ocr": ocr_result,
            "template_match": match_public,
            "authenticity": authenticity_result,
            "integration_decision": decision,
            "visuals": {
                "aligned": png_data_url(
                    match_result.get("aligned_bgr"),
                    rgb=False,
                ),
                "match_visualization": png_data_url(
                    match_visual,
                    rgb=True,
                ),
                "template_preview": png_data_url(
                    template_preview,
                    rgb=True,
                ),
            },
        }
        return jsonify(response)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print(f"OCR directory: {OCR_DIR.resolve()}")
    print(f"Templates: {TEMPLATES_DIR.resolve()}")
    print(f"Templates loaded: {len(gallery.templates)} ({gallery.matcher_name})")
    print(f"Authenticity model: {AUTHENTICITY_MODEL_PATH}")
    print(f"Authenticity scoring available: {authenticity_scorer.available}")
    app.run(host="127.0.0.1", port=8787, debug=False)