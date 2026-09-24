# Sijil + Mustakhrij OCR + Single-Request Template Verification

The browser sends each selected document once to the local Python API. The
backend calls Mustakhrij OCR through Ollama, compares the image with the
included reference gallery using OpenCV, and scores it with a pre-trained
genuine/fake classifier. Template similarity and the authenticity score are
both structural/statistical signals; neither alone proves a document is
genuine.

This is a local demo, not a production identity-verification system. Do not
expose the backend to the internet or use its output as the sole basis for an
identity decision. The authenticity score comes from `authenticity_clf.joblib`,
trained on genuine vs. synthetically-tampered examples - treat it as a
research signal, not a certified forensic verdict.

## After Ollama is installed and the model is downloaded

The model used by the app is `qwen3-vl:4b`. Confirm that Ollama can see it:

```bash
ollama list
```

If `qwen3-vl:4b` is not listed, download it once:

```bash
ollama pull qwen3-vl:4b
```

Ollama usually stays running as a background service on Windows and macOS. If
it is not running on your system, start `ollama serve` in a separate terminal
and leave that terminal open. Do not start a second Ollama server if the
background service is already running.

## Install the Python dependencies

Open a terminal **in this project folder** (the folder containing
`sijil_verification_backend.py`). Build a fresh virtual environment; do not
reuse a `.venv` copied from another computer or operating system.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the app

1. In terminal 1, from the project folder with the virtual environment active,
   start the API:

   ```bash
   python sijil_verification_backend.py
   ```

   Leave it running. It loads the templates at startup and serves the API on
   `http://127.0.0.1:8787`.

2. In terminal 2, change to the same project folder and start the static page
   server:

   ```bash
   python -m http.server 8000
   ```

3. Open this address in your browser:

   ```text
   http://127.0.0.1:8000/Sijil-integrated-single-request.html
   ```

4. Choose a JPG, JPEG, PNG, or WEBP document image. The local API accepts
   uploads up to 12 MB and limits decoded image size to 40 megapixels.

Optional API check (open in a browser or run with curl):

```text
http://127.0.0.1:8787/api/health
```

The response should show `"ok": true`, a nonzero `template_count`, the
selected matcher (`SIFT` or `ORB`), and `"authenticity_model_loaded": true`.
If the model failed to load, `authenticity_model_error` explains why; the
OCR and template-match flow still work normally in that case.

By default the backend looks for the classifier at
`fakepassports/authenticity_clf.joblib`. Override the location with an
environment variable if needed:

```bash
export SIJIL_AUTHENTICITY_MODEL=/path/to/authenticity_clf.joblib
```

On macOS/Linux, `bash start_sijil.sh` can start the API and page server
together from one terminal. Stop it with Ctrl+C.

## Command-line test

With the virtual environment active and Ollama running:

```bash
python document_integration.py Mustakhrij-OCR/test.jpeg
```

The optional Gradio interface needs the extra package:

```bash
python -m pip install gradio
python document_integration.py --gradio
```

The separate `fakepassports/run_locally.py` research demo also uses the
included classifier and needs these optional packages:

```bash
python -m pip install gradio joblib scikit-learn
python fakepassports/run_locally.py
```

Its classifier score is a research signal, not a forensic conclusion.

## Troubleshooting

- **Ollama connection refused:** start the Ollama desktop service or run
  `ollama serve` in another terminal.
- **Model not found:** run `ollama list`; if needed, run
  `ollama pull qwen3-vl:4b`.
- **The API says no templates were found:** keep the included
  `fakepassports/templates/` folder beside the backend; run the backend from
  this project folder.
- **`authenticity_model_loaded` is `false`:** confirm
  `fakepassports/authenticity_clf.joblib` exists and that `joblib` and
  `scikit-learn` are installed (`pip install -r requirements.txt` covers
  both). Check `authenticity_model_error` in `/api/health` for the exact
  reason.
- **`InconsistentVersionWarning` when the backend starts:** the classifier
  was saved with a different scikit-learn version than the one installed
  locally. It is usually safe to ignore, but if scores look wrong, install
  a matching scikit-learn version or re-export the classifier from the
  training notebook with your current version.
- **The browser cannot reach port 8787:** make sure the backend terminal is
  still running. The page must be opened from the local server on port 8000
  (or directly as a `file://` page).
- **Upload rejected:** use one of the supported image formats, under 12 MB and
  no larger than 40 megapixels.
- **First request is slow:** Ollama may be loading Qwen into memory; later
  requests are usually faster.