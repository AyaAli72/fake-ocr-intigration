# OCR Formal Documents

Extracts fields from official documents (passports, national ID cards,
residence permits, driver's licenses, etc.) in any language, using a local
vision-language model — fully offline, no cloud API calls.

## How it works

Sends the document image to a local **Qwen3-VL-4B** model (via Ollama) with
a prompt describing the JSON shape to return. The model identifies the
document type and reads out whatever fields it can, in one step — no
per-country templates needed.

Currently stops at raw JSON output for manual review. Validation, confidence
scoring, and the save/flag decision are the next step once the output looks
solid.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) — runs the model locally, not a pip package:
  - **macOS**: `brew install ollama`
  - **Debian/Ubuntu**: see [ollama.com/download](https://ollama.com/download)
  - **Windows**: install from [ollama.com/download](https://ollama.com/download)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama and pull the model (one-time, ~3GB download):

```bash
brew services start ollama    # or just run `ollama serve` in a terminal
ollama pull qwen3-vl:4b
```

Verify it's ready:

```bash
ollama list
```

You should see `qwen3-vl:4b` in the list.

## Usage

```bash
python extract.py path/to/document.jpg
```

Prints the model's raw JSON output: document type, issuing country, and
whatever fields it read. Check it against the actual document by eye —
there's no automated validation yet.
