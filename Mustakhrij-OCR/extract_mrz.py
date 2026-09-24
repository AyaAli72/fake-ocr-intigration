import sys
import json
from passporteye import read_mrz


def extract(image_path: str) -> dict:
    mrz = read_mrz(image_path)
    if mrz is None:
        raise ValueError(
            f"Could not locate an MRZ in {image_path!r}. "
            "Try a sharper, well-lit, non-glare photo of just the bio page."
        )
    data = mrz.to_dict()
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_mrz.py <path-to-passport-image>")
        sys.exit(1)

    result = extract(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result.get("valid_score", 0) == 100:
        print(
            f"\nWarning: MRZ checksum confidence is {result.get('valid_score')}/100 "
            "— re-scan the image if this is below 100.",
            file=sys.stderr,
        )
