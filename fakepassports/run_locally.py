"""
Passport / ID Nationality Matcher + Authenticity Scoring — LOCAL VERSION
=========================================================================

Run this on your own machine, no Kaggle/GPU needed.

Setup:
    pip install opencv-python scikit-learn joblib pillow numpy gradio

Folder layout expected (next to this script):
    ./templates/                <- unzip templates.zip here (one .jpg per nationality/type)
    ./authenticity_clf.joblib   <- downloaded from Kaggle Output panel

Run:
    python run_locally.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import os
import io
import glob
import cv2
import joblib
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "authenticity_clf.joblib")


# ---------------------------------------------------------------------------
# 1. Rebuild the ORB template gallery from the exported template images
# ---------------------------------------------------------------------------
def load_gray(path, max_dim=900):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class TemplateGallery:
    def __init__(self, n_features=2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.templates = {}

    def add_template(self, name, image_path):
        gray = load_gray(image_path)
        if gray is None:
            print(f"  [WARN] could not read {image_path}")
            return
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            print(f"  [WARN] too few keypoints for {name}, skipping")
            return
        self.templates[name] = {"gray": gray, "kp": kp, "des": des, "path": image_path}

    def build_from_dir(self, directory):
        paths = sorted(glob.glob(os.path.join(directory, "*.jpg")))
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            self.add_template(name, p)
        print(f"Built gallery with {len(self.templates)} templates from {directory}")


# ---------------------------------------------------------------------------
# 2. Forensic feature extraction (must match the notebook exactly)
# ---------------------------------------------------------------------------
def ela_map(img_bgr, quality=90):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb.astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = np.array(Image.open(buf).convert("RGB")).astype(np.int16)
    orig = np.array(im.convert("RGB")).astype(np.int16)
    return np.abs(orig - resaved).mean(axis=2)


def block_stats(ela, block=16):
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
    final = cv2.imdecode(cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
    ela = ela_map(final, quality=90)
    z, block_std, block_mean = block_stats(ela)
    lap_var = cv2.Laplacian(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return [z, block_std, block_mean, lap_var, ela.mean(), ela.std()]


# ---------------------------------------------------------------------------
# 3. Load everything once at startup
# ---------------------------------------------------------------------------
print("Loading template gallery...")
gallery = TemplateGallery(n_features=2000)
gallery.build_from_dir(TEMPLATES_DIR)

print("Loading trained authenticity classifier...")
authenticity_clf = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# 4. Inference function (identical logic to the notebook's Gradio demo)
# ---------------------------------------------------------------------------
def identify_and_visualize(image, min_inliers):
    if image is None:
        return "Please upload an image.", None, None
    if len(gallery.templates) == 0:
        return "No templates found — check the templates/ folder.", None, None

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    scale = 900 / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    query_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    q_kp, q_des = gallery.orb.detectAndCompute(query_gray, None)
    if q_des is None or len(q_kp) < 10:
        return "Could not detect enough features. Try a clearer scan.", None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ratio_thresh = 0.75
    results = []
    good_by_name = {}

    for name, tpl in gallery.templates.items():
        matches = bf.knnMatch(q_des, tpl["des"], k=2)
        good = [
            first
            for pair in matches
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < ratio_thresh * second.distance
        ]
        good_by_name[name] = good

        inliers = 0
        if len(good) >= 4:
            src_pts = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([tpl["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())

        results.append({"name": name, "inliers": inliers, "good_matches": len(good)})

    results.sort(key=lambda r: r["inliers"], reverse=True)
    best = results[0]
    confident = best["inliers"] >= min_inliers

    feats = np.array([extract_forensic_features(img_bgr)])
    fake_proba = authenticity_clf.predict_proba(feats)[0, 1]
    genuine_pct = (1 - fake_proba) * 100
    fake_pct = fake_proba * 100

    if genuine_pct >= 70:
        verdict = "Likely GENUINE"
    elif genuine_pct >= 40:
        verdict = "UNCERTAIN - manual review recommended"
    else:
        verdict = "Likely FAKE / tampered"

    lines = [
        "### Nationality / Type match",
        f"**Predicted:** {best['name'] if confident else 'UNKNOWN / no confident match'}",
        f"Inlier matches: {best['inliers']}  |  Good matches: {best['good_matches']}  |  Confident: {'Yes' if confident else 'No'}",
        "",
        "### Authenticity check",
        f"**Verdict: {verdict}**",
        f"Genuine probability: **{genuine_pct:.1f}%**  |  Fake/tampered probability: **{fake_pct:.1f}%**",
        "*(Heuristic estimate from a classifier trained on genuine vs. synthetically-tampered "
        "examples — treat as a research signal, not a certified forensic verdict.)*",
        "",
        "**Top nationality candidates:**",
    ]
    for r in results[:5]:
        lines.append(f"- {r['name']}: inliers={r['inliers']}, good_matches={r['good_matches']}")
    summary = "\n".join(lines)

    best_name = best["name"]
    tpl = gallery.templates[best_name]
    good = good_by_name[best_name]
    vis = cv2.drawMatches(
        query_gray, q_kp, tpl["gray"], tpl["kp"], good[:40], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(tpl["gray"], cv2.COLOR_GRAY2RGB)

    return summary, vis_rgb, template_rgb


# ---------------------------------------------------------------------------
# 5. Local Gradio UI (same layout as the notebook, minus share=True)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(title="Passport Nationality Matcher (Local)") as demo:
        gr.Markdown(
            """
            # Passport / ID Nationality Matcher — Local
            Upload a scanned passport/ID. Compares it against the local template
            gallery (ORB keypoint matching + RANSAC homography) and scores authenticity
            with the pre-trained classifier.

            *Uses mock/synthetic identity documents (MIDV500) for research/educational use only.*
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload scanned passport/ID", type="numpy")
                threshold_slider = gr.Slider(4, 50, value=10, step=1, label="Confidence threshold (min inliers)")
                run_btn = gr.Button("Match", variant="primary")
            with gr.Column():
                result_text = gr.Markdown()
        with gr.Row():
            match_vis = gr.Image(label="Keypoint matches (query <-> best template)")
            template_preview = gr.Image(label="Matched reference template")

        run_btn.click(fn=identify_and_visualize,
                      inputs=[image_input, threshold_slider],
                      outputs=[result_text, match_vis, template_preview])

    # share=False -> runs purely on your machine, no public tunnel
    demo.launch(share=False, debug=False)
