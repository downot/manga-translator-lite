# Manga Image Translator Lite

[English](README.md) | [日本語](README-JP.md) | [中文](README-CN.md)

## Acknowledgements

This project is deeply indebted to **frederik-uni** and **zyddnys** and the original [manga-image-translator](https://github.com/zyddnys/manga-image-translator). Without [frederik-uni](https://github.com/frederik-uni) and [zyddnys](https://github.com/zyddnys)' outstanding work, this project would not exist. This "Lite" version is a simplified, modularized, and modernized refactor of the original codebase. The core capabilities of this project will still be based on the original project.

## Core Differences from Original

1.  **Decoupled Pipeline**: Unlike the monolithic or web-service approach of the original, Lite splits the process into three distinct CLI steps: `extract`, `translate`, and `render`. Intermediate results are stored in a human-readable `pages.json`, allowing manual review or editing before the final render.
2.  **LLM Batching Optimization**: Specifically redesigned for Large Language Models. It batches text blocks across multiple pages to significantly reduce API costs and provide better context for translation.
3.  **Modernized & Optimized**: Fully compatible with Python 3.9+ (fixing type-hinting issues in the original) and optimized for Apple Silicon (MPS/Metal acceleration).
4.  **Smart Rendering**: Features a binary-search font fitting algorithm that automatically maximizes font size to fill bubble areas while respecting the original detected boundaries, resulting in cleaner and more professional-looking typeset.
5.  **Simplified Dependencies**: Removed heavy web UI and backend components to focus on a lean, CLI-first experience that is easier to integrate into automation scripts.

---

Local OCR + third-party LLM API. The pipeline is split into three reviewable
steps so you can edit translations by hand before they get rendered back onto
the page.

```
  input/                                work_dir/                          out/
  ┌───────────┐                         ┌─────────────┐                    ┌───────────┐
  │ 0001.jpg  │ ─── extract (CV) ──▶  │ pages.json  │ ── render ──▶     │ 0001.png  │
  │ 0002.jpg  │                       │ clean/0001  │                    │ 0002.png  │
  └───────────┘                         └─────────────┘                    └───────────┘
                                              ▲
                                              │ translate (LLM API)
                                              │ + manual edit
```

## Steps

| Step | What it does | Outputs |
|---|---|---|
| `extract` | text detection → OCR → mask refinement → inpainting | `work_dir/clean/*.png`, `work_dir/pages.json` (text + positions) |
| `translate` | groups blocks into ~1500-char batches, calls the configured LLM, fills `translation` fields | updated `pages.json` |
| `render` | paints translations onto the inpainted images | `out_dir/*.png` |
| `run` | extract → translate → render in one shot | both |

`pages.json` is the single source of truth. Open it between `translate` and
`render` to revise any translation.

## Quick start

```bash
pip install -r requirements.txt          # python >= 3.10, < 3.12
cp examples/Example.env .env             # add OPENAI_API_KEY or GEMINI_API_KEY

python -m manga_translator_lite extract -i ./input -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work -c examples/config-example.toml
python -m manga_translator_lite render ./work -o ./out -c examples/config-example.toml

# Or end-to-end (skips manual review)
python -m manga_translator_lite run -i ./input -w ./work -o ./out -c examples/config-example.toml
```

## Configuration

A single TOML or JSON file. All sections are optional; defaults are sensible.

```toml
use_gpu = false

[detector]
detector = "default"        # default | dbconvnext | ctd | craft | paddle | none
detection_size = 2048

[ocr]
ocr = "48px"                # 32px | 48px | 48px_ctc | mocr
min_text_length = 0

[inpainter]
inpainter = "lama_large"    # default | lama_large | lama_mpe | none
inpainting_size = 2048

[translator]
provider = "openai"          # openai | gemini | none
model = "gpt-4o-mini"
api_base = "https://api.openai.com/v1"
target_lang = "ENG"
batch_chars = 1500           # ~1000–2000 chars per LLM request
context_pages = 1            # number of past pages sent as tone context

[render]
font_size_offset = 0
direction = "auto"
alignment = "auto"
```

`provider = "openai"` covers any OpenAI-compatible HTTP endpoint, including
DeepSeek, OpenRouter, Groq and Ollama — point `api_base` and `model` at the
service you want.

API keys can live in `[translator] api_key` or in env vars
(`OPENAI_API_KEY` / `GEMINI_API_KEY`); see [examples/Example.env](examples/Example.env).

Print the full schema with:

```bash
python -m manga_translator_lite config-help
```

## Editing translations

After `translate`, `pages.json` looks like:

```json
{
  "version": 1,
  "target_lang": "ENG",
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
          "lines": [...],
          "font_size": 24,
          "direction": "auto",
          "alignment": "auto"
        }
      ]
    }
  ]
}
```

Edit any `translation` field, then run `render`.

## Project layout

```
manga_translator_lite/
  pipeline/        extract / translate / render
  detection/       text detection (default/dbconvnext/ctd/craft/paddle)
  ocr/             local OCR models (32px / 48px / 48px_ctc / manga-ocr)
  textline_merge/  group lines into text blocks
  mask_refinement/ refine the inpaint mask around text
  inpainting/      AOT, LaMa-large, LaMa-MPE
  translators/     unified LLM client (OpenAI-compatible + Gemini)
  rendering/       paint translated text back onto the clean image
  utils/           shared helpers, model wrappers, logging
```

## Usage

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

Configure your API key by following:

```bash
cp examples/Example.env .env

# Modify the .env file with your API key and other settings
```

Add a folder named `input` in the root directory, place your manga images in it, and run:

```bash
python -m manga_translator_lite extract -i ./input -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
