# OCR + Template Matching Integration

This integration keeps `Mustakhrij-OCR` unchanged.

## Why this architecture

`Mustakhrij-OCR/extract.py` is responsible for reading text and returning:

- `document_type`
- `issuing_country`
- extracted fields
- `photo_box`

The template layer consumes that JSON and uses it only to narrow the candidate
templates. It then performs visual matching with:

1. document quadrilateral alignment;
2. SIFT, with ORB fallback;
3. Lowe ratio filtering;
4. RANSAC homography;
5. inlier count, inlier ratio, and reprojection error.

This is stronger than matching against every template by raw ORB inlier count,
and it avoids asking the OCR model a second time through `verify_document.py`.

## Setup

The combined project archive already contains the OCR module and the 50
template images under `fakepassports/templates/`; there is no separate
templates archive to extract. From the project root, install the shared
dependencies:

```text
workspace/
  document_integration.py
  Mustakhrij-OCR/
    extract.py
    requirements.txt
  fakepassports/
    templates/
      01_alb_id.jpg
      02_aut_drvlic_new.jpg
      ...
```

```bash
python -m pip install -r requirements.txt
```

Ollama must be running with the `qwen3-vl:4b` model available. Follow the root
`README.md` for Windows, macOS, and Linux startup steps.

## CLI test

```bash
python document_integration.py path/to/document.jpg \
  --ocr-dir ./Mustakhrij-OCR \
  --templates-dir ./fakepassports/templates
```

The output contains three explainable sections:

- `ocr`: the unchanged OCR response and validation status;
- `template_match`: best template, candidates, inliers, and geometric score;
- `integration_decision`: whether OCR type agrees with the visual template.

## Gradio UI

```bash
python document_integration.py --gradio \
  --ocr-dir ./Mustakhrij-OCR \
  --templates-dir ./fakepassports/templates
```

Use `--share` only when a temporary public link is needed:

```bash
python document_integration.py --gradio --share
```

## Important interpretation

A template match means that the document layout resembles a known reference.
It does not prove authenticity. Keep the authenticity classifier and MRZ
validation as separate signals; do not let template matching silently override
OCR or claim forensic certainty.