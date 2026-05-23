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

![Editor Screenshot](screenshot.jpg)
*Example of the Visual Editor (editor.html) in action.*

- **Real-time Preview**: See how the translated text looks on the actual page.
- **Quick Edit**: Modify translations in a sidebar and see instant updates on canvas.
- **Keyboard Shortcuts**: `←`/`→` for paging, `Z` for zoom, `R` for reload, `S` for save.
- **Shortcut Focus Routing**: Intelligently routes Up/Down arrow keys based on active focus:
  - **Tasks Sidebar**: Switch between different manga tasks.
  - **Pages Sidebar**: Flip through page indexes.
  - **Dialogue Editor**: Navigate text blocks while editing textareas.
  - **Canvas Viewer**: Scroll the page vertically when zoomed in.
- **Scrollable Sidebar**: Support long task lists gracefully with independent sidebar flex scrolling.

### How to Use:
There are two ways to open the editor:
1. **Serverless Local Mode**: Open `editor.html` directly in your browser. Click **"Open Work Dir"** to select your `work` folder. (Requires Chrome/Edge, uses the modern HTML5 File System Access API for local reads/writes).
2. **Standalone Server Mode**: Run `python server.py -w ./work` and visit `http://localhost:8000/editor.html`.

---

## Standalone Backend Server (`server.py`)

In addition to serving static files, this project includes a standalone backend server (`server.py`) to manage task data, synchronize story descriptions, and execute pipelines directly from your browser.

### 1. Key Features
* **Web-based Pipeline Runner**: Trigger `Extract`, `Translate`, `Render`, or the full sequential pipeline (`Run`) directly from the visual editor browser page, with live terminal logs streamed back via Server-Sent Events (SSE).
* **Automated Data Syncing**: Save page data, manual translation edits, and story contexts (`story.txt`) directly to your disk, bypassing standard browser security sandboxing.
* **Token-based Security**: Automatically generates robust access tokens for each task directory to prevent unauthorized data manipulation.

### 2. Usage
Run the server pointing to your active workspace:
```bash
python server.py -w ./work -p 8000
```
Then visit `http://localhost:8000/editor.html` in your browser. The console will display secure URLs for each task in your workspace.

---

## Translation Review & Story Context Management

To address inconsistency in tone and lack of context when translating long or continuous manga chapters in batches, this project introduces the **Story-Context-Aware Translation Review and Polish (Review)** feature during the `translate` stage.

### 1. Mechanism & Idempotency
* **Review & Polish**: After completing the initial translation, the pipeline automatically compiles all translated blocks in chronological reading order, then invokes the LLM to polish them based on the **overall story description** for consistency and character voice.
* **Idempotency**: A `<lang>.reviewed` marker file (e.g. `CHS.reviewed`) is created in the `translations/` folder upon a successful review. Subsequent `translate` runs will skip both translation and review.
* **Incremental Run**: If a task is fully translated but not yet reviewed, running `translate` will skip the translation phase and **incrementally run the review step**; using the `--overwrite` flag forces both re-translation and re-review.

### 2. Story Context File (`story.txt`)
Simply place a text file named `story.txt` (or `script.txt`, `description.txt`) in your task folder (e.g. `work/task_a/`). The program will automatically search for it. You can also specify it in `pages.json` under the `"story"` key.
Write character bios, relationship details, and style/tone notes in it. The LLM will leverage this outline to perform highly immersive, consistent translations.

### 3. Visual Story Editor & Prompts
Story context management has been fully integrated into the **Visual Editor (`editor.html`)**:
* **New Story Tab**: A sidebar "Story" tab has been added to write, edit, and save `story.txt` on the fly.
* **Genre-based Templates**: Built-in multi-lingual (EN/ZH/JA) genre templates (Daily Romance, Fantasy Adventure, Gag/Comedy, Mature/Adult 18+) to bootstrap context writing.
* **Serverless Local Writing**: Utilizing the HTML5 File System Access API, it saves `story.txt` directly to your local workspace disk without needing any active backend server process. It also automatically synchronizes via API when running in `server.py` mode.

### 4. Anti-Censorship Disclaimer
To prevent LLMs (e.g. DeepSeek, Gemini) from rejecting adult or mature manga during translation, a standard English disclaimer ("All characters depicted in the work are entirely fictional and over 18 years old...") is embedded in the system-level prompts, ensuring a smooth translation pipeline.

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
