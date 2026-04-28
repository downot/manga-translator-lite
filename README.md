# Manga Image Translator Lite

[English](README.md) | [日本語](README-JP.md) | [中文](README-CN.md)

## Acknowledgements

This project is deeply indebted to **frederik-uni** and **zyddnys** and the original [manga-image-translator](https://github.com/zyddnys/manga-image-translator). This "Lite" version is a modularized and modernized refactor aimed at providing a high-performance, CLI-first experience with human-in-the-loop flexibility.

## Core Differences from Original

1.  **Decoupled Pipeline**: Splits the process into `extract`, `translate`, and `render`. Intermediate results are stored in `pages.json`, allowing manual review or editing before the final render.
2.  **LLM Batching Optimization**: Specifically redesigned for Large Language Models. It batches text blocks across multiple pages to significantly reduce API costs and provide better context for translation.
3.  **Modernized & Optimized**: Fully compatible with Python 3.10+ and optimized for Apple Silicon (MPS/Metal) and NVIDIA (CUDA) acceleration.
4.  **Smart Rendering**: Features a binary-search font fitting algorithm that automatically maximizes font size to fill bubble areas while respecting the original detected boundaries.
5.  **Multi-Task Support**: Automatically handles multiple manga folders as separate "tasks", keeping a clean workspace structure.
6.  **Incremental Translation**: Supports resuming from a specific page and skipping already translated blocks to save time and API costs.

---

Local OCR + third-party LLM API. The pipeline is split into three reviewable steps so you can edit translations by hand before they get rendered back onto the page.

```text
  in/                                   work/                              out/
  ├── manga_a/                          ├── manga_a/                       ├── manga_a/
  │   ├── 0001.jpg  ── extract ──▶    │   ├── pages.json  ── render ──▶ │   ├── 0001.png
  │   └── 0002.jpg                     │   └── clean/                     │   └── 0002.png
  └── manga_b/                          └── manga_b/                       └── manga_b/
      ├── 0001.jpg                          ├── pages.json                     ├── 0001.png
      └── 0002.jpg                          └── clean/                         └── 0002.png
                                                  ▲
                                                  │ translate (LLM API)
                                                  │ + manual edit (Visual Editor)
```

Each subdirectory under `in/` is treated as a separate **task**. The directory structure is mirrored into `work/` and `out/`. Images where no text is detected are passed through unchanged.

## Steps

| Step | What it does | Outputs |
|---|---|---|
| `extract` | text detection → OCR → mask refinement → inpainting | `work/<task>/clean/*.png`, `work/<task>/pages.json` |
| `translate` | batches blocks (~1500 chars), calls LLM, fills `translation` fields. Supports incremental updates. | updated `pages.json` per task |
| `render` | paints translations onto the inpainted images using smart typesetting | `out/<task>/*.png` (same count as input) |
| `run` | extract → translate → render in one shot | Both workspace and final images |

Each task's `pages.json` is the single source of truth. Open it between `translate` and `render` to revise any translation.

## Quick start

```bash
pip install -r requirements.txt          # Python >= 3.10
cp examples/Example.env .env             # Add OPENAI_API_KEY or GEMINI_API_KEY

# Single command end-to-end
python -m manga_translator_lite run -i ./in -w ./work -o ./out

# Or step-by-step
python -m manga_translator_lite extract -i ./in -w ./work
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## Configuration

A single TOML or JSON file. All sections are optional; defaults are sensible.

```toml
use_gpu = true

[detector]
detector = "default"        # Options: default | dbconvnext | ctd | craft | paddle
detection_size = 2048

[ocr]
ocr = "48px"                # Options: 32px | 48px | 48px_ctc | mocr

[translator]
provider = "openai"          # Options: openai | gemini
model = "gpt-4o-mini"
api_base = "https://api.openai.com/v1"
target_lang = "ENG"
batch_chars = 1500           # ~1000–2000 chars per LLM request
context_pages = 2            # number of past pages sent as tone context

[render]
font_size_offset = 0
direction = "auto"           # Options: auto | horizontal | vertical
alignment = "auto"
```

`provider = "openai"` covers any OpenAI-compatible HTTP endpoint, including DeepSeek, Groq and Ollama. API keys can live in `[translator] api_key` or in `.env` vars (`OPENAI_API_KEY` / `GEMINI_API_KEY`).

## Visual Editor (Experimental)

A lightweight web-based visual editor `editor.html` is provided for a better manual review experience.

- **Real-time Preview**: See how the translated text looks on the actual page.
- **Quick Edit**: Modify translations in a sidebar and see instant updates on canvas.
- **Keyboard Shortcuts**: `←`/`→` for paging, `Z` for zoom, `R` for reload, `S` for save.

### How to Use:
1. Start a local server: `python -m http.server 8000`
2. Open `http://localhost:8000/editor.html` in Chrome/Edge.
3. Click **"Open Work Dir"** and select your `work` folder.

## Editing translations

After `translate`, each task's `pages.json` looks like:

```json
{
  "version": 2,
  "target_lang": "ENG",
  "task_name": "manga_a",
  "pages": [
    {
      "index": 0,
      "name": "0001.jpg",
      "size": [1200, 1700],
      "clean": "clean/0000_0001.png",
      "blocks": [
        {
          "id": "p0000_b000",
          "text": "おはよう",
          "translation": "Good morning",
          "bbox": [120, 340, 80, 40],
          "polygon": [[120,340],[200,340],[200,380],[120,380]],
          "font_size": 24
        }
      ]
    },
    {
      "index": 1,
      "name": "0002.jpg",
      "no_text": true,
      "blocks": []
    }
  ]
}
```

### Re-translating & Resuming

Lite supports smart incremental updates and resuming:

```bash
# Re-translate from index 10 onwards
python -m manga_translator_lite translate ./work --start-index 10

# Force re-translate everything (overwrites existing translations)
python -m manga_translator_lite translate ./work --overwrite
```

## Project layout

```text
manga_translator_lite/
  pipeline/        # Core CLI steps (extract, translate, render, run)
  translators/     # Unified LLM clients (OpenAI-compatible, Gemini)
  rendering/       # Smart typesetting and font fitting
  detection/       # Text detection modules
  ocr/             # Local OCR wrappers
  ...
```

## Usage

Recommended setup using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp examples/Example.env .env  # Configure your API keys
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
