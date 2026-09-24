import sys
import json
import ollama
from PIL import Image

MODEL = "qwen3-vl:4b"

PROMPT = """You are reading an official document (passport, national ID card, \
residence permit, driver's license, or similar). The document may be in any \
language.

First identify the document type — regardless of what language the document \
itself is written in, classify it into exactly one of these fixed English / \
Arabic pairs, and output that pair exactly as shown:
- Passport / جواز سفر
- National ID Card / بطاقة هوية وطنية
- Residence Permit / تصريح إقامة
- Driver's License / رخصة قيادة
- Birth Certificate / شهادة ميلاد
- Other / أخرى  (only if none of the above fit)

Then extract every field you can read, in English field names but \
original-language values (e.g. keep names as printed).

Every single field, with no exceptions, must also carry your confidence in \
that reading:
- "high": clearly printed, no ambiguity
- "medium": legible but something makes it uncertain (unusual font, glare, \
partial obstruction, an odd transliteration)
- "low": blurry, partially obscured, or you are essentially guessing

photo_box is REQUIRED in every response, with no exceptions: find the \
bounding box of the MAIN, LARGEST portrait photograph of the document \
holder (ignore any small secondary ghost/watermark photo elsewhere on the \
document, e.g. overlapping the document number), as fractions of the full \
image width/height (0.0 to 1.0), x/y being the top-left corner. Only omit \
photo_box if the document truly has zero photo anywhere on it.

Return ONLY a JSON object, no other text, in this shape — every field value \
is an object with "value" and "confidence", never a bare string:
{
  "document_type": "Passport / جواز سفر",
  "issuing_country": "...",
  "photo_box": {"x": 0.05, "y": 0.15, "width": 0.25, "height": 0.35},
  "fields": {
    "<field name>": {"value": "...", "confidence": "high"},
    "<field name>": {"value": "...", "confidence": "medium"}
  }
}
If a field is not present or not legible, omit it rather than guessing."""


def extract(image_path: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_path],
        }],
        format="json",
        think=False,
        keep_alive="30m",
    )
    message = response["message"]
    # This Ollama build of Qwen3-VL sometimes puts the answer in `thinking`
    # instead of `content`, even with think=False. Fall back to whichever
    # is non-empty.
    return message["content"] or message.get("thinking", "")


def validate(record: dict) -> dict:
    """Decide whether a record is safe to save or needs human review,
    based on the per-field confidence the model reported."""
    fields = record.get("fields")
    if not record.get("document_type") or not fields:
        return {"status": "FLAG", "reason": "missing document_type or fields"}

    low, medium, unrated = [], [], []
    for name, f in fields.items():
        if not isinstance(f, dict) or "confidence" not in f:
            unrated.append(name)
        elif f["confidence"] == "low":
            low.append(name)
        elif f["confidence"] == "medium":
            medium.append(name)

    if low or unrated:
        reasons = []
        if low:
            reasons.append(f"low confidence: {', '.join(low)}")
        if unrated:
            reasons.append(f"no confidence reported: {', '.join(unrated)}")
        return {"status": "FLAG", "reason": "; ".join(reasons)}

    if medium:
        return {"status": "SAVE", "reason": f"medium confidence, worth a glance: {', '.join(medium)}"}
    return {"status": "SAVE", "reason": "all fields high confidence"}


def pad_box(box: dict, margin: float = 0.15) -> dict:
    """Expand a normalized (0-1) bounding box by a margin on each side,
    clamped to stay within the image. The model's box is sometimes a
    little too tight and clips part of the face — a margin compensates."""
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    dx, dy = w * margin, h * margin
    nx, ny = max(0.0, x - dx), max(0.0, y - dy)
    nw = min(1.0 - nx, w + 2 * dx)
    nh = min(1.0 - ny, h + 2 * dy)
    return {"x": nx, "y": ny, "width": nw, "height": nh}


def save_photo(image_path: str, box: dict, out_path: str) -> bool:
    """Crop the ID photo out of the source image using a normalized
    (0-1) bounding box and save it. Returns False if the box looks invalid."""
    try:
        box = pad_box(box)
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0):
            return False
        img = Image.open(image_path)
        iw, ih = img.size
        crop = (
            int(x * iw), int(y * ih),
            min(iw, int((x + w) * iw)), min(ih, int((y + h) * ih)),
        )
        img.crop(crop).save(out_path)
        return True
    except (KeyError, TypeError, ValueError, OSError):
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract.py <path-to-document-image>")
        sys.exit(1)

    raw = extract(sys.argv[1])
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        print("STATUS: FLAG")
        print("REASON: model did not return valid JSON\n")
        print(raw)
        sys.exit(0)

    decision = validate(record)
    print(f"STATUS: {decision['status']}")
    print(f"REASON: {decision['reason']}\n")

    if record.get("photo_box"):
        photo_path = sys.argv[1].rsplit(".", 1)[0] + "_photo.png"
        if save_photo(sys.argv[1], record["photo_box"], photo_path):
            print(f"PHOTO: saved to {photo_path}\n")
        else:
            print("PHOTO: model returned a photo_box but it looked invalid, skipped\n")

    print(json.dumps(record, indent=2, ensure_ascii=False))
