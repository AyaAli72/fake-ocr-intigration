"""
OCR + template matching integration.

This file deliberately does not modify Mustakhrij-OCR.

Pipeline:
    image
      -> existing Mustakhrij-OCR extract.py
      -> parsed document_type / issuing_country
      -> OCR-guided template candidates
      -> perspective alignment
      -> SIFT (or ORB fallback) + Lowe ratio test
      -> RANSAC homography and inlier scoring
      -> structured result for review/UI

Expected folders after unzipping the two projects:

    ./Mustakhrij-OCR/extract.py
    ./fakepassports/templates/*.jpg

CLI:
    python document_integration.py path/to/document.jpg \
        --ocr-dir ./Mustakhrij-OCR \
        --templates-dir ./fakepassports/templates

Optional Gradio UI:
    python document_integration.py --gradio \
        --ocr-dir ./Mustakhrij-OCR \
        --templates-dir ./fakepassports/templates
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_ocr_module(ocr_dir: str | Path):
    """Import the existing OCR module without changing its source code."""
    ocr_dir = Path(ocr_dir).resolve()
    if not (ocr_dir / "extract.py").exists():
        raise FileNotFoundError(f"extract.py was not found in {ocr_dir}")

    if str(ocr_dir) not in sys.path:
        sys.path.insert(0, str(ocr_dir))
    return importlib.import_module("extract")


def run_existing_ocr(image_path: str | Path, ocr_dir: str | Path) -> dict:
    """Call Mustakhrij-OCR exactly as it is, then parse and validate its JSON."""
    ocr_module = load_ocr_module(ocr_dir)
    raw = ocr_module.extract(str(image_path))

    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "status": "FLAG",
            "reason": "OCR did not return valid JSON",
            "raw": raw,
            "record": {},
        }

    decision = ocr_module.validate(record)
    return {
        "status": decision.get("status", "FLAG"),
        "reason": decision.get("reason", ""),
        "record": record,
    }


def read_bgr(image_or_path):
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_or_path}")
        return image

    image = np.asarray(image_or_path)
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def warp_quad(image, points):
    rect = order_points(points)
    tl, tr, br, bl = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def align_document(bgr):
    """Crop a large document quadrilateral; return the original on failure."""
    height, width = bgr.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = (
        cv2.resize(bgr, None, fx=scale, fy=scale)
        if scale < 1
        else bgr.copy()
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = small.shape[0] * small.shape[1]
    best = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.20 * image_area or area <= best_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            best = polygon.reshape(4, 2).astype(np.float32)
            best_area = area

    if best is None:
        return bgr
    if scale < 1:
        best = best / scale
    return warp_quad(bgr, best)


def document_kind(document_type: str) -> str | None:
    text = str(document_type or "").lower()
    if "passport" in text or "جواز" in text:
        return "passport"
    if "driver" in text or "license" in text or "رخص" in text:
        return "drvlic"
    if "id" in text or "national" in text or "هوية" in text:
        return "id"
    if "permit" in text or "إقامة" in text:
        return "permit"
    return None


COUNTRY_ALIASES = {
    "albania": "alb",
    "austria": "aut",
    "azerbaijan": "aze",
    "brazil": "bra",
    "chile": "chl",
    "china": "chn",
    "czech": "cze",
    "czechia": "cze",
    "germany": "deu",
    "denmark": "dnk",
    "algeria": "dza",
    "spain": "esp",
    "estonia": "est",
    "finland": "fin",
    "greece": "grc",
    "croatia": "hrv",
    "hungary": "hun",
    "iran": "irn",
    "italy": "ita",
    "japan": "jpn",
    "latvia": "lva",
    "moldova": "mda",
    "macau": "mac",
    "norway": "nor",
    "poland": "pol",
    "portugal": "prt",
    "romania": "rou",
    "russia": "rus",
    "serbia": "srb",
    "slovakia": "svk",
    "turkey": "tur",
    "ukraine": "ukr",
    "uruguay": "ury",
    "united states": "usa",
    "usa": "usa",
}


def country_code(value: str) -> str | None:
    text = str(value or "").strip().lower()
    if len(text) == 3 and text.isalpha():
        return text
    for name, code in COUNTRY_ALIASES.items():
        if name in text:
            return code
    return None


## ---------------------------------------------------------------------------
## Authenticity scoring (genuine vs. synthetically-tampered classifier)
##
## Feature extraction here is copied EXACTLY from fakepassports/run_locally.py
## (which itself matches the training notebook) so a query image produces the
## same 6-number feature vector the classifier was actually trained on. Do not
## change this math without retraining authenticity_clf.joblib to match.
## ---------------------------------------------------------------------------
def _ela_map(img_bgr, quality=90):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb.astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = np.array(Image.open(buf).convert("RGB")).astype(np.int16)
    orig = np.array(im.convert("RGB")).astype(np.int16)
    return np.abs(orig - resaved).mean(axis=2)


def _block_stats(ela, block=16):
    h, w = ela.shape
    means = []
    for y in range(0, max(h - block, 1), block):
        for x in range(0, max(w - block, 1), block):
            means.append(ela[y:y + block, x:x + block].mean())
    means = np.array(means) if means else np.array([0.0])
    med = np.median(means)
    mad = np.median(np.abs(means - med)) + 1e-6
    z = (means.max() - med) / mad
    return z, means.std(), means.mean()


def extract_forensic_features(img_bgr):
    final = cv2.imdecode(
        cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1],
        cv2.IMREAD_COLOR,
    )
    ela = _ela_map(final, quality=90)
    z, block_std, block_mean = _block_stats(ela)
    lap_var = cv2.Laplacian(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return [z, block_std, block_mean, lap_var, ela.mean(), ela.std()]


class AuthenticityScorer:
    """Wraps the pre-trained genuine/fake classifier (authenticity_clf.joblib).

    Loading is best-effort: if the model file or the optional joblib/
    scikit-learn packages are missing, `available` is False and callers
    should keep the rest of the OCR/template pipeline working normally.
    This mirrors the project's existing philosophy of never letting one
    signal silently block another.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.clf = None
        self.load_error: str | None = None

        if not self.model_path.exists():
            self.load_error = f"Model file not found: {self.model_path}"
            return
        try:
            import joblib
        except ImportError:
            self.load_error = (
                "joblib is not installed. Run: pip install joblib scikit-learn"
            )
            return
        try:
            self.clf = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001 - surface any load failure, not just ImportError
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.clf is not None

    def score(self, bgr_image) -> dict:
        if not self.available:
            return {
                "available": False,
                "error": self.load_error,
            }

        height, width = bgr_image.shape[:2]
        scale = 900.0 / max(height, width)
        image = (
            cv2.resize(bgr_image, (int(width * scale), int(height * scale)))
            if scale < 1
            else bgr_image
        )

        features = np.array([extract_forensic_features(image)])
        fake_probability = float(self.clf.predict_proba(features)[0, 1])
        genuine_percent = round((1.0 - fake_probability) * 100.0, 1)
        fake_percent = round(fake_probability * 100.0, 1)

        if genuine_percent >= 70:
            verdict = "LIKELY_GENUINE"
        elif genuine_percent >= 40:
            verdict = "UNCERTAIN_REVIEW"
        else:
            verdict = "LIKELY_FAKE"

        return {
            "available": True,
            "genuine_percent": genuine_percent,
            "fake_percent": fake_percent,
            "verdict": verdict,
            "note": (
                "Heuristic estimate from a classifier trained on genuine vs. "
                "synthetically-tampered examples - a research signal, not a "
                "certified forensic verdict."
            ),
        }


class TemplateGallery:
    """Template matcher with OCR-guided candidate filtering and RANSAC scoring."""

    def __init__(self, templates_dir: str | Path, n_features: int = 3000):
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates folder not found: {self.templates_dir}")

        if hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.norm = cv2.NORM_L2
            self.matcher_name = "SIFT"
            self.ratio_threshold = 0.72
        else:
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.norm = cv2.NORM_HAMMING
            self.matcher_name = "ORB"
            self.ratio_threshold = 0.75

        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        paths = sorted(
            path
            for path in self.templates_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 8:
                continue
            self.templates[path.stem] = {
                "path": str(path),
                "gray": gray,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }

        if not self.templates:
            raise RuntimeError(f"No usable templates found in {self.templates_dir}")

    def candidate_names(self, ocr_record: dict) -> list[str]:
        names = list(self.templates.keys())
        kind = document_kind(ocr_record.get("document_type", ""))
        if kind:
            filtered = [
                name
                for name in names
                if kind in name.lower()
                or (kind == "id" and "_id" in name.lower())
            ]
            if filtered:
                names = filtered

        code = country_code(
            ocr_record.get("issuing_country")
            or ocr_record.get("nationality")
            or ""
        )
        if code:
            country_filtered = [name for name in names if f"_{code}_" in f"_{name.lower()}_"]
            if country_filtered:
                names = country_filtered
        return names

    def _match_one(self, query_gray, query_keypoints, query_descriptors, name):
        template = self.templates[name]
        pairs = cv2.BFMatcher(self.norm).knnMatch(
            query_descriptors,
            template["descriptors"],
            k=2,
        )
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.ratio_threshold * second.distance
        ]

        inliers = 0
        reprojection_error = None
        if len(good) >= 4:
            src = np.float32(
                [query_keypoints[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst = np.float32(
                [template["keypoints"][m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())
                if inliers:
                    projected = cv2.perspectiveTransform(src, homography)
                    errors = np.linalg.norm(
                        projected[inlier_mask, 0] - dst[inlier_mask, 0],
                        axis=1,
                    )
                    reprojection_error = float(errors.mean())

        inlier_ratio = inliers / max(len(good), 1)
        # The score rewards both enough inliers and geometric consistency.
        score = (
            0.55 * min(inliers / 40.0, 1.0)
            + 0.45 * inlier_ratio
        )
        if reprojection_error is not None:
            score *= float(np.clip(1.0 - reprojection_error / 20.0, 0.0, 1.0))

        return {
            "name": name,
            "inliers": inliers,
            "good_matches": len(good),
            "inlier_ratio": round(inlier_ratio, 4),
            "reprojection_error": reprojection_error,
            "score": round(float(score), 4),
            "matches": good,
        }

    def match(self, image_or_array, ocr_record: dict | None = None) -> dict:
        original_bgr = read_bgr(image_or_array)
        aligned_bgr = align_document(original_bgr)
        query_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        query_keypoints, query_descriptors = self.detector.detectAndCompute(
            query_gray,
            None,
        )
        if query_descriptors is None or len(query_keypoints) < 8:
            return {
                "status": "NO_FEATURES",
                "reason": "Not enough visual features for template matching.",
                "matcher": self.matcher_name,
                "candidates": [],
                "aligned_bgr": aligned_bgr,
            }

        ocr_record = ocr_record or {}
        candidates = self.candidate_names(ocr_record)
        results = [
            self._match_one(
                query_gray,
                query_keypoints,
                query_descriptors,
                name,
            )
            for name in candidates
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        best = results[0]

        # These thresholds are intentionally conservative. A match is not
        # called genuine merely because it is the best among weak candidates.
        confident = (
            best["inliers"] >= 12
            and best["inlier_ratio"] >= 0.25
            and best["score"] >= 0.25
        )
        return {
            "status": "MATCHED" if confident else "REVIEW",
            "matcher": self.matcher_name,
            "best": {
                key: value
                for key, value in best.items()
                if key != "matches"
            },
            "top_candidates": [
                {
                    key: value
                    for key, value in result.items()
                    if key != "matches"
                }
                for result in results[:5]
            ],
            "candidate_count": len(candidates),
            "aligned_bgr": aligned_bgr,
            "_query_keypoints": query_keypoints,
            "_query_gray": query_gray,
            "_best_matches": best["matches"],
        }

    def visualization(self, match_result: dict):
        if match_result.get("status") == "NO_FEATURES":
            return None, None
        best_name = match_result["best"]["name"]
        template = self.templates[best_name]
        vis = cv2.drawMatches(
            match_result["_query_gray"],
            match_result["_query_keypoints"],
            template["gray"],
            template["keypoints"],
            match_result["_best_matches"][:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        template_rgb = cv2.cvtColor(template["gray"], cv2.COLOR_GRAY2RGB)
        return vis_rgb, template_rgb


def run_pipeline(image_path, ocr_dir, templates_dir, authenticity_model_path=None):
    """Run unchanged OCR first, then use its result to guide matching."""
    ocr_result = run_existing_ocr(image_path, ocr_dir)
    record = ocr_result.get("record", {})

    gallery = TemplateGallery(templates_dir)
    match_result = gallery.match(image_path, record)

    # Remove internal arrays/keypoints before returning JSON.
    public_match = {
        key: value
        for key, value in match_result.items()
        if not key.startswith("_") and key not in {"aligned_bgr"}
    }

    authenticity = None
    if authenticity_model_path:
        scorer = AuthenticityScorer(authenticity_model_path)
        authenticity = scorer.score(read_bgr(image_path))

    return {
        "ocr": ocr_result,
        "template_match": public_match,
        "authenticity": authenticity,
        "integration_decision": integration_decision(
            ocr_result,
            public_match,
            authenticity,
        ),
    }


def integration_decision(
    ocr_result: dict,
    match_result: dict,
    authenticity: dict | None = None,
) -> dict:
    """Keep the signals explainable instead of silently overriding OCR."""
    ocr_record = ocr_result.get("record", {})
    best = match_result.get("best", {})
    ocr_kind = document_kind(ocr_record.get("document_type", ""))
    template_kind = document_kind(best.get("name", ""))
    type_agrees = bool(ocr_kind and template_kind and ocr_kind == template_kind)
    matched = match_result.get("status") == "MATCHED"

    if matched and type_agrees:
        verdict = "TEMPLATE_MATCH_CONFIRMED"
    elif matched:
        verdict = "TEMPLATE_MATCH_REVIEW_OCR_TYPE"
    elif match_result.get("status") == "NO_FEATURES":
        verdict = "REVIEW_NO_TEMPLATE_FEATURES"
    else:
        verdict = "REVIEW_WEAK_TEMPLATE_MATCH"

    decision = {
        "verdict": verdict,
        "ocr_status": ocr_result.get("status", "FLAG"),
        "ocr_document_type": ocr_record.get("document_type"),
        "matched_template": best.get("name"),
        "template_score": best.get("score"),
        "document_type_agrees": type_agrees,
        "note": (
            "Template agreement is a structural signal, not proof that the "
            "document is genuine. Keep the existing OCR output unchanged."
        ),
    }

    # Authenticity is reported as an independent signal alongside the
    # template/OCR verdict above, never merged into it - a strong template
    # match says "this looks like a real passport layout", which is a
    # different question from "was this specific image tampered with".
    if authenticity is not None:
        decision["authenticity_available"] = authenticity.get("available", False)
        if authenticity.get("available"):
            decision["authenticity_verdict"] = authenticity.get("verdict")
            decision["authenticity_genuine_percent"] = authenticity.get("genuine_percent")
        else:
            decision["authenticity_error"] = authenticity.get("error")

    return decision


def build_gradio_demo(ocr_dir, templates_dir, authenticity_model_path=None):
    import gradio as gr

    gallery = TemplateGallery(templates_dir)
    scorer = AuthenticityScorer(authenticity_model_path) if authenticity_model_path else None

    def predict(image):
        if image is None:
            return "Upload an image.", None, None

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            bgr = read_bgr(image)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                tmp.name,
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            ocr_result = run_existing_ocr(tmp.name, ocr_dir)

        record = ocr_result.get("record", {})
        match_result = gallery.match(image, record)
        public_match = {
            key: value
            for key, value in match_result.items()
            if not key.startswith("_") and key not in {"aligned_bgr"}
        }
        authenticity = scorer.score(read_bgr(image)) if scorer else None
        decision = integration_decision(ocr_result, public_match, authenticity)
        vis, template = gallery.visualization(match_result)

        summary = json.dumps(
            {
                "ocr": ocr_result,
                "template_match": public_match,
                "authenticity": authenticity,
                "integration_decision": decision,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        return summary, vis, template

    with gr.Blocks(title="OCR + Template Verification") as demo:
        gr.Markdown(
            """
            # OCR + Template Verification

            The existing OCR remains unchanged. Its document type and country
            guide the template search, then SIFT/ORB + RANSAC checks structural
            agreement with the reference gallery.
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="numpy",
                    label="Upload document image",
                )
                run_button = gr.Button("Run verification", variant="primary")
            result = gr.Code(label="OCR + template result", language="json")
        with gr.Row():
            match_visualization = gr.Image(label="Geometric matches")
            matched_template = gr.Image(label="Best template")

        run_button.click(
            predict,
            inputs=image_input,
            outputs=[result, match_visualization, matched_template],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--ocr-dir", default="./Mustakhrij-OCR")
    parser.add_argument("--templates-dir", default="./fakepassports/templates")
    parser.add_argument(
        "--authenticity-model",
        default="./fakepassports/authenticity_clf.joblib",
        help="Path to the trained genuine/fake classifier. Pass an empty "
             "string to skip authenticity scoring entirely.",
    )
    parser.add_argument("--gradio", action="store_true")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    auth_path = args.authenticity_model or None

    if args.gradio:
        demo = build_gradio_demo(args.ocr_dir, args.templates_dir, auth_path)
        demo.launch(share=args.share, debug=False)
        return

    if not args.image:
        parser.error("Provide an image path or use --gradio.")
    result = run_pipeline(
        args.image,
        args.ocr_dir,
        args.templates_dir,
        auth_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()