"""
Structural layout verification for identity documents.

Idea: a country's passport/ID design changes over time (a new "version"
every several years). Given the document's own nationality + issue date, we
know which version *should* apply, and can check whether key elements
(photo, MRZ, name field, document number field) sit where that version's
real specimens put them. A field in the wrong place is a red flag.

Honest limits, worth keeping in mind: this is a coarse structural sanity
check based on a single reference specimen per country/version, not a full
forensic fraud analysis. It cannot see UV ink, IR features, holograms, or
chip data — those need specialized scanning hardware, not a phone photo.
It also inherits noise from the model's own zone detection, which varies
run to run by several percentage points even on the same image — that's
why the comparison below uses a generous tolerance rather than an exact
match.
"""
import sys
import json
from datetime import date
import ollama

MODEL = "qwen3-vl:4b"

PROMPT = """You are examining an identity document (passport, ID card, \
residence permit) for fraud verification. The document may be in any \
language.

Extract these two fields, REQUIRED in every response with no exceptions:
- nationality: the nationality/issuing country shown on the document, in \
English
- issueDate: the date of issue, in ISO 8601 format (YYYY-MM-DD). If only \
the issue year is visible, use YYYY-01-01.

Also locate the bounding boxes of these structural elements. CRITICAL: \
every coordinate MUST be a normalized fraction between 0.0 and 1.0 \
(fraction of image width/height), NEVER a raw pixel number. photo, mrz, \
and nameField are REQUIRED in every response with no exceptions, unless \
truly absent from the document:
- photo: the main portrait photograph
- mrz: the machine-readable zone (monospace lines with < characters, \
usually at the bottom)
- nameField: the printed name field/label area
- docNumberField: the printed document/passport number field area (omit \
only if you cannot find it)

Return ONLY JSON in this shape:
{"nationality":"...","issueDate":"YYYY-MM-DD","zones":{"photo":{"x":0.05,"y":0.20,"width":0.25,"height":0.35},"mrz":{"x":0.05,"y":0.85,"width":0.9,"height":0.12},"nameField":{},"docNumberField":{}}}"""


# Reference zones measured from real specimens (see /test*.jpeg). Each
# nationality/doc type has a list of versions; version selection uses the
# document's own issue date, so this can grow to multiple eras per country
# as more reference specimens are added.
#
# Deliberately NOT sourced from our own test/demo images (test1-5.jpeg) --
# those are for testing the pipeline against, and measuring a reference
# from the same data you test against would be circular. Zones below were
# measured from a real specimen independently published on Wikimedia
# Commons (used to illustrate Wikipedia's "Saudi passport" article), whose
# biodata page matches the description of the current biometric design
# (green cover, e-passport chip symbol, redesigned data page). Face and
# most personal fields in that source image were already blurred by the
# original uploader; only its structural layout (element positions) was
# measured here, no personal data was extracted or stored.
TEMPLATE_REGISTRY = {
    ("saudi arabia", "passport"): [
        {
            "version": "2021+ biometric e-passport",
            "valid_from": "2021-01-01",  # earliest confirmed: source specimen issued 2021-03-18
            "valid_to": None,
            "source": "https://commons.wikimedia.org/wiki/File:Annotation_2022-02-11_222351-1.jpg "
                      "(independent specimen, face/name/number redacted by original uploader; "
                      "geometry only was measured, no personal data used)",
            "zones": {
                "photo": {"x": 0.04, "y": 0.29, "width": 0.23, "height": 0.35},
                "mrz": {"x": 0.03, "y": 0.75, "width": 0.97, "height": 0.15},
                "nameField": {"x": 0.29, "y": 0.28, "width": 0.35, "height": 0.06},
                "docNumberField": {"x": 0.79, "y": 0.14, "width": 0.19, "height": 0.08},
            },
        },
    ],
}


def extract_for_verification(image_path: str) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [image_path]}],
        format="json",
        think=False,
    )
    message = response["message"]
    raw = message["content"] or message.get("thinking", "")
    return json.loads(raw)


def zone_center(zone: dict) -> tuple[float, float]:
    return (zone["x"] + zone["width"] / 2, zone["y"] + zone["height"] / 2)


def point_in_padded_zone(point: tuple[float, float], zone: dict, padding: float = 0.4) -> bool:
    """True if point falls inside `zone` expanded by `padding` fraction of
    its own size on each side. Generous on purpose — see module docstring."""
    px, py = point
    dx, dy = zone["width"] * padding, zone["height"] * padding
    x0, y0 = zone["x"] - dx, zone["y"] - dy
    x1, y1 = zone["x"] + zone["width"] + dx, zone["y"] + zone["height"] + dy
    return x0 <= px <= x1 and y0 <= py <= y1


def resolve_template(nationality: str, doc_type: str, issue_date: str) -> tuple[dict | None, str]:
    """Returns (template, reason). template is None if nothing matched;
    reason distinguishes "we don't cover this country/type at all" from
    "we cover it, but not for this issue date" -- different failure modes,
    worth surfacing differently rather than one generic message."""
    key = (nationality.strip().lower(), doc_type)
    versions = TEMPLATE_REGISTRY.get(key)
    if not versions:
        return None, f"no reference template at all for nationality={nationality!r}, doc_type={doc_type!r}"
    for v in versions:
        lo, hi = v["valid_from"], v["valid_to"]
        if lo and issue_date < lo:
            continue
        if hi and issue_date > hi:
            continue
        return v, "matched"
    covered = ", ".join(f"{v['version']} ({v['valid_from'] or '...'} to {v['valid_to'] or '...'})" for v in versions)
    return None, f"issue date {issue_date!r} is outside the date range of every version we have on file ({covered})"


def verify(image_path: str, doc_type: str = "passport") -> dict:
    data = extract_for_verification(image_path)
    nationality = data.get("nationality", "")
    issue_date = data.get("issueDate", "")
    detected = data.get("zones", {})

    template, reason = resolve_template(nationality, doc_type, issue_date)
    if template is None:
        return {
            "nationality": nationality,
            "issueDate": issue_date,
            "verdict": "UNKNOWN",
            "confidence": None,
            "reason": reason,
            "fields": {},
        }

    expected = template["zones"]
    field_results = {}
    for field, exp_zone in expected.items():
        det_zone = detected.get(field)
        if not det_zone:
            field_results[field] = {"status": "MISSING", "ok": False}
            continue
        ok = point_in_padded_zone(zone_center(det_zone), exp_zone)
        field_results[field] = {"status": "OK" if ok else "OUT_OF_POSITION", "ok": ok, "detected": det_zone}

    checked = list(field_results.values())
    passed = sum(1 for f in checked if f["ok"])
    confidence = round(100 * passed / len(checked)) if checked else None

    if confidence is None:
        verdict = "UNKNOWN"
    elif confidence >= 75:
        verdict = "LIKELY_GENUINE"
    elif confidence >= 50:
        verdict = "REVIEW_RECOMMENDED"
    else:
        verdict = "SUSPICIOUS"

    return {
        "nationality": nationality,
        "issueDate": issue_date,
        "templateVersion": template["version"],
        "verdict": verdict,
        "confidence": confidence,
        "fields": field_results,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_document.py <path-to-document-image>")
        sys.exit(1)

    result = verify(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
