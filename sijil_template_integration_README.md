# Sijil template verification integration

This adds visual template verification to the existing Sijil HTML without
changing the Mustakhrij-OCR prompt or OCR implementation.

## Files

- `sijil_verification_backend.py`: local Flask API.
- `sijil_template_integration.js`: UI panel injected into the existing Sijil registration page.
- `document_integration.py`: OCR + SIFT/ORB + RANSAC integration.

## Folder layout

```text
workspace/
  Mustakhrij-OCR/
    extract.py
  fakepassports/
    templates/
      01_alb_id.jpg
      ...
  document_integration.py
  sijil_verification_backend.py
  sijil_template_integration.js
  Sijil — Standalone HTML (AR EN, Offline).html
```

Extract `fakepassports/templates.zip` into `fakepassports/templates/`.

## Start the backend

```bash
pip install flask opencv-python numpy
python sijil_verification_backend.py
```

The API runs at:

```text
http://127.0.0.1:8787
```

The OCR project still needs Ollama and `qwen3-vl:4b` running locally.

## Add the panel to Sijil

Before the closing `</body>` tag in the Sijil HTML, add:

```html
<script src="./sijil_template_integration.js"></script>
```

If the JavaScript file is elsewhere:

```html
<script src="./path/to/sijil_template_integration.js"></script>
```

The integrated HTML sends the selected image to the local backend once. The
backend runs the unchanged OCR once, performs template matching, scores the
image with the pre-trained genuine/fake classifier, and emits one combined
result back to the page. After an upload, the page adds a card showing:

- OCR document type and issuing country;
- best matching template;
- geometry score;
- RANSAC inliers;
- inlier ratio;
- aligned document;
- feature-match visualization;
- reference template;
- top candidate templates;
- authenticity check: genuine/fake percentage and verdict from
  `authenticity_clf.joblib`, shown as its own section so it is never
  confused with the template-match verdict above it.

The browser does not call Ollama directly in this enhanced version. This
prevents duplicate OCR calls and keeps Ollama, OpenCV, and the classifier behind
one local API.

## Important

Open the HTML through a local HTTP server rather than relying on `file://`:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/Sijil%20%E2%80%94%20Standalone%20HTML%20%28AR%20EN%2C%20Offline%29.html
```

The browser page remains a UI. Ollama, OCR, and OpenCV template matching stay
in the local Python backend.