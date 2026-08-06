# Manga Image Translator Lite

[English](README.md) | [日本語](README-JP.md) | [中文](README-CN.md)

## Acknowledgements

This project is deeply indebted to **frederik-uni** and **zyddnys** and the original [manga-image-translator](https://github.com/zyddnys/manga-image-translator). This "Lite" version is a modularized and modernized refactor aimed at providing a high-performance, CLI-first experience with human-in-the-loop flexibility.

## Core Differences from Original

1.  **Decoupled Pipeline**: Splits the process into `extract`, `translate`, and `render`. Intermediate results are stored in `pages.json`, allowing manual review or editing before the final render.
2.  **Context-safe LLM Translation**: Translates page by page, retaining only real preceding-page context and optional same-page approved text; prompt budgets keep long story notes from crowding out source lines.
3.  **Modernized & Optimized**: Fully compatible with Python 3.10+ and optimized for Apple Silicon (MPS/Metal) and NVIDIA (CUDA) acceleration.
4.  **Smart Rendering**: Features a binary-search font fitting algorithm that automatically maximizes font size to fill bubble areas while respecting the original detected boundaries.
5.  **Multi-Task Support**: Automatically handles multiple manga folders as separate "tasks", keeping a clean workspace structure.
6.  **Incremental Translation**: Supports resuming from a specific page and skipping already translated blocks to save time and API costs.
7.  **Spelling & Fluency Proofreader**: Built-in copyediting stage using LLM before rendering to check and correct typos, bad grammar, and translation awkwardness.
8.  **Preserved Newline Layout**: Fully supports explicit newlines (`\n`) for dialog formatting, ensuring preview rendering on browser matches Python typesetting perfectly.

---

Local OCR + third-party LLM API. The pipeline is split into reviewable stages so you can edit translations by hand and optionally run a separate whole-work review before rendering.

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
| `translate` | batches blocks within each page (~1500 chars), calls the LLM, and fills `translation` fields. Supports incremental updates. | updated `pages.json` per task |
| `review` | optionally polish existing translations with the LLM; never runs automatically. | reviewed translation files and marker |
| `render` | paints translations onto the inpainted images using smart typesetting. Optional copyediting/proofreading check (--check) before rendering. | `out/<task>/*.png` (same count as input) |
| `run` | extract → translate → render in one shot | Both workspace and final images |

Each task's `pages.json` is the single source of truth. Open it between `translate` and `render` to revise any translation.

## Quick start

```bash
pip install -r requirements.txt          # Python >= 3.10
cp config.toml.sample config.toml        # Then fill in [translator] api_key (or use an env var)
cp .env.sample .env             # Optional: OPENAI_API_KEY or GEMINI_API_KEY instead

# Single command end-to-end
python -m manga_translator_lite run -i ./in -w ./work -o ./out

# Or step-by-step
python -m manga_translator_lite extract -i ./in -w ./work
python -m manga_translator_lite translate ./work
# Optional, manual whole-work review
python -m manga_translator_lite review ./work
python -m manga_translator_lite render ./work -o ./out
```

## Command Reference

All commands are invoked as `python -m manga_translator_lite <command> [options]`.

**Common options** (available on `extract`, `translate`, `review`, `render`, `run`):

| Option | Description |
|---|---|
| `-c, --config <path>` | Path to the `.toml`/`.json` config file. Defaults to `./config.toml` (or `config.json`) when present. |
| `--target-lang <code>` | Override `translator.target_lang` for this run only — `config.toml` is **not** modified (e.g. `CHS`, `ENG`, `JPN`, `KOR`). |
| `-v, --verbose` | Verbose logging and intermediate diagnostics. |

### `extract` — Step 1: detection / OCR / inpaint → workspace

| Argument | Description |
|---|---|
| `-i, --input <path>` *(required)* | Input image file, or a directory of images / sub-task folders. |
| `-w, --work-dir <path>` *(required)* | Workspace directory to create or update. |
| `--overwrite` | Re-extract every image even if already present; existing translations are salvaged by spatial IoU matching. |

### `translate` — Step 2: call the LLM and fill in translations

| Argument | Description |
|---|---|
| `work_dir` *(positional, required)* | Existing workspace directory. |
| `--overwrite` | Re-translate blocks that already have translations. Hand-edited blocks (`edited: true`) are preserved. |
| `--start-index <n>` | Start (re)translating from this page index; earlier pages are used as context only. |
| `--reference-lang <code>` | Reference an already-translated language as a semantic/tone hint (repeatable). Omit for **auto** (every other human-reviewed language). See [Cross-language reference](#cross-language-reference-translation). |
| `--no-reference` | Disable cross-language reference; translate purely from the source. |
| `-j, --concurrency <n>` | Translate this many **tasks** in parallel (overrides `[translator] concurrency`). A single task is never split, so each keeps its full cross-page context — concurrency only runs several independent works at once. `1` = sequential (default). For cloud LLMs, `3–5` cuts wall-clock time; stay under your provider's rate limits. Local/GPU models gain little. |

> **Per-language output:** `--target-lang` temporarily overrides the configured language without touching `config.toml`. Output is keyed by language code (`translations/CHS.json`, `translations/ENG.json`, …), so the same workspace can hold several languages side by side — translate once per language, no overwriting:
>
> ```bash
> python -m manga_translator_lite translate ./work --target-lang CHS
> python -m manga_translator_lite translate ./work --target-lang JPN
> ```

### `review` — Optional manual whole-work translation polish

`translate` never changes completed translations through the review model. Run review explicitly when a full-work consistency pass is desired:

```bash
python -m manga_translator_lite review ./work --target-lang CHS
```

`--overwrite` reviews a language again even when its `translations/<LANG>.reviewed` marker exists. This command is best used after a human has checked terminology and source OCR, since it can still change voice or editorial wording.

### `render` — Step 3: paint translations onto clean images

| Argument | Description |
|---|---|
| `work_dir` *(positional, required)* | Existing workspace directory. |
| `-o, --output <path>` *(required)* | Output directory for final images. |
| `--check` | Force the spelling/fluency proofreading check before rendering. |
| `--no-check` | Skip the proofreading check entirely. |
| `-y, --yes` | Auto-accept and apply all proofreading suggestions. |

### `run` — extract + translate + render end-to-end

Accepts the union of the options above: `-i/--input`, `-w/--work-dir`, `-o/--output` *(all required)*, plus `--overwrite`, `--check`, `--no-check`, `-y/--yes`, and the translate-phase `--reference-lang`/`--no-reference`.

### `config-help` — print the JSON schema of the config file

```bash
python -m manga_translator_lite config-help
```

### Editor server (`server.py`)

| Option | Default | Description |
|---|---|---|
| `-w, --work-dir <path>` | `work` | Workspace directory to serve. |
| `-p, --port <n>` | `8000` | Port to listen on. |
| `--host <addr>` | `0.0.0.0` | Bind address (use `127.0.0.1` to restrict to localhost). |
| `--log-file <path>` | `server.log` | Path to the server log file. |

---

## Configuration

The pipeline reads a single TOML (or JSON) file. The easiest start is to copy the fully-annotated sample and edit it:

```bash
cp config.toml.sample config.toml   # then fill in [translator] api_key
```

`config.toml` is git-ignored, so your key never gets committed. See **[config.toml.sample](config.toml.sample)** for every option with inline comments. All sections are optional; defaults are sensible. A minimal example:

```toml
use_gpu = true

[detector]
detector = "ctd"            # Options: ctd | default | dbconvnext | craft | paddle | rtdetr | none
detection_size = -1         # -1 = auto per page; use 2048/2560 for a fixed global size
detection_size_scale = 1.0  # auto mode: raise to 1.3–1.6 for tiny/dense text
detection_size_min = 1024
detection_size_max = 2560
text_threshold = 0.25
box_threshold = 0.6
unclip_ratio = 1.8
secondary_detector = "none" # set "rtdetr" to fuse in a high-recall box detector

[ocr]
ocr = "48px"                # Options: 32px | 48px | 48px_ctc | mocr

[translator]
provider = "openai"          # Options: openai | gemini | none
model = "gpt-4o-mini"
api_base = ""                # Empty = provider default, or e.g. https://openrouter.ai/api/v1
api_key = ""                 # Or leave empty and set the OPENAI_API_KEY env var
target_lang = "ENG"
source_lang = "auto"        # source-language hint, or let the model detect each line
profile = "manga"           # manga | magazine | general
batch_chars = 1500           # ~1000–2000 chars per LLM request
context_pages = 1            # number of past pages sent as tone context
temperature = 0.3            # lower = more consistent names/register; higher = more varied
use_vision = false           # attach page images for vision-capable LLMs (slower/costlier)
vision_max_pages_per_batch = 1
prompt_token_budget = 6000   # cap story/context prompt growth
story_context_token_budget = 1200
context_token_budget = 900
concurrency = 1              # task-level parallelism; use 3–5 for cloud LLMs
# reference_langs unset = auto (reference every human-reviewed language); [] = off;
# ["CHS"] = reference exactly these. Referenced languages are read-only.

[render]
font_path = "fonts/GenEiAntiqueNv5-M.ttf"
font_size_offset = 0
font_size_minimum = 34       # Lower bound so small text stays legible
font_size_minimum_expand_limit = 2.5  # Max box growth allowed to host the minimum font size
font_size_readable_min = -1  # auto readability floor for fixed/user-sized boxes
line_spacing = 0             # Tighten line spacing so more text fits at a larger size
direction = "auto"           # Options: auto | horizontal | vertical
alignment = "auto"
disable_font_border = false  # Keep the outline — key for legibility on any background
# font_color = "000000:FFFFFF"  # Black text + white outline; good for pure B/W books only

[signature]
enabled = false              # set true and fill translator to bake a credit
translator = ""
pages = "first_last"         # none | first | last | first_last | every
```

`provider = "openai"` covers any OpenAI-compatible HTTP endpoint, including DeepSeek, OpenRouter, Groq and Ollama. API keys can live in `[translator] api_key` or in `.env` vars (`OPENAI_API_KEY` / `GEMINI_API_KEY`).

### Choosing an OCR engine

`[ocr] ocr` controls recognition only; it does not change text detection. The `32px` / `48px` names are the recognizer's normalized text height, not `detection_size`. Start with `48px`; changing OCR requires re-running `extract --overwrite` to regenerate blocks.

| Value | Best use | Trade-off |
|---|---|---|
| `"32px"` | Fast, lightweight extraction on CPU / constrained VRAM, or clean large print where throughput matters most. | Legacy 32-pixel autoregressive model; more likely to miss tiny, dense, or decorative text. |
| `"48px"` | Default for mixed-language comics, magazines, and most general pages. Best first baseline for ordinary horizontal or vertical text. | A little more compute than `32px`, but usually noticeably better recall on small text. |
| `"48px_ctc"` | Try when `48px` makes systematic character mistakes, or when using `ocr.ignore_bubble` to skip free/SFX text outside balloons. | Alternative CTC decoder, not a universal accuracy upgrade; compare a few representative pages before adopting it for a whole task. |
| `"mocr"` | Japanese-dominant manga, especially handwritten or stylized Japanese that the bundled recognizers read poorly. Set `use_mocr_merge = true` only when multi-line Japanese regions should be recognized as one phrase. | Slowest option: manga-ocr runs per region in addition to the normal geometry/color pass. Merging can combine adjacent lines, so inspect the resulting blocks before translating. |

The sample config is intentionally conservative; a common high-recall setup for mixed page sizes is `detection_size = -1`, `secondary_detector = "rtdetr"`, `secondary_box_threshold = 0.25–0.3`, and `concurrency = 3` for hosted LLMs. Keep secrets out of shared configs: leave `api_key = ""` and use environment variables when committing or sending examples.

### Choosing a text detector · RT-DETR

The `[detector] detector` option is pluggable: `default`/`dbconvnext` (DBNet), `ctd` (Comic Text Detector), `craft`, `paddle`, `none`, and the experimental **`rtdetr`**.

`rtdetr` wraps the Apache-2.0 Hugging Face model `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2 r50vd), which detects bubbles / in-bubble text / free text and is good at region typing and grouping on stylized or webtoon/manhua pages. Caveats:

- Needs `transformers` (`pip install transformers`); torch is already a dependency. It is loaded lazily, so the rest of the package is unaffected if you don't use this detector.
- It is a **box detector** — it returns rectangles, not stroke-level masks, so its erase mask is coarser than DBNet/CTD. Keep `ctd`/`default` for production inpainting; treat `rtdetr` as a detection / region-typing experiment for now.
- Try a lower confidence: set `[detector] box_threshold ≈ 0.3`.
- Apache-2.0 covers the code/weights, **not** the (undisclosed) training data — verify provenance before commercial use.

RT-DETR shines when it catches regions your primary detector misses on stylized / webtoon / SFX-heavy pages. Rather than switching detectors wholesale (and losing stroke-level erase masks), you can **fuse** it in — keep a mask-producing detector (ctd/default) as primary and add RT-DETR's extra regions on top.

#### Fusing two detectors (`secondary_detector`)

If a second detector catches regions your primary misses, you don't have to choose — **fuse** them. Set a `[detector] secondary_detector` and the extract step runs both: it keeps your primary detector's stroke-level masks for clean erasing, then **adds the secondary's regions that the primary missed** (IoU below `fusion_iou`) to detection. Extra regions are OCR'd, but become translation blocks only after extract derives a safe local stroke mask; when it cannot do so, the region is skipped instead of drawing a translation over source text. Set `secondary_box_fill = true` only to restore the legacy whole-box behavior.

```toml
[detector]
detector = "ctd"                # primary — keep a stroke detector for clean masks
secondary_detector = "rtdetr"   # booster — adds the regions ctd misses (needs `transformers`)
secondary_box_threshold = 0.3   # rtdetr likes a lower confidence than ctd/dbnet
fusion_iou = 0.4                # a secondary region is "new" only if it overlaps no primary box above this
fusion_overlap_limit = 0.5      # also drop it as a duplicate if it covers/contains a primary box this much (see below)
fusion_max_area_ratio = 0.1     # drop secondary boxes larger than this fraction of the page (see below)
```

`secondary_detector = "none"` (the default) disables fusion entirely — behavior is unchanged. This is purely additive recall: the primary still owns the mask, so erase quality is the primary's everywhere except the extra regions.

**Avoiding duplicate extraction.** A box detector returns large region-level boxes, while a stroke detector returns small per-line boxes — so a secondary box sitting *on top of* primary text lines has a low IoU and would slip through as "new", getting OCR'd a second time (often as a partial copy). `fusion_overlap_limit` also rejects a secondary box when it covers, or is covered by, that fraction of any primary box (intersection over the smaller box), so such overlaps are dropped before OCR. Lower it (e.g. `0.3`) if the same text still comes out twice.

**Avoiding large false erasures.** A box detector can return one huge box for a stylized title or SFX that spans the artwork. `fusion_max_area_ratio` drops any secondary box covering more than that fraction of the page (default `0.1` = 10%), so oversized region boxes are neither translated nor erased — the art is left intact. Lower it (e.g. `0.06`) if uncertain large regions remain; set `0` to disable the cap.

#### VRAM & speed

Detection and inpainting alternate between large, differently-sized tensors per page (e.g. `detection_size` 2560 then `inpainting_size` 2048), which fragments CUDA memory. A few built-in measures keep high-recall settings (a large `detection_size` **and** a secondary detector) within budget:

- **RT-DETR runs in fp16 on CUDA** automatically — the log shows `RT-DETR running in fp16 (CUDA)` — roughly halving the secondary detector's VRAM with no change to recall (CPU/MPS stay fp32).
- **VRAM is released between the detect and inpaint stages** of every page, so their peak allocations don't stack and don't fragment across pages.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by default** (in the package init, before torch initializes its allocator) to curb fragmentation. Export your own value to override it; on Windows / older torch builds that don't support it, torch simply ignores it.

Tuning levers, from safest to most quality-affecting:

- **`det_gamma_correct`** — keep it **off** for normal (bright, high-contrast) manga. It auto-darkens bright pages toward mid-grey, which *lowers* detection/erase quality and adds per-page work; it only helps genuinely dark / washed-out scans.
- **`inpainting_size`** (e.g. 2048 → 1536) — trims the inpaint stage's VRAM with little visible effect on erased text; the most VRAM-for-quality-friendly knob.
- **`detection_size`** and **`secondary_detector`** — your strongest recall levers, so lower/disable these *last*: shrinking `detection_size` or turning the secondary detector off cuts VRAM but visibly drops recognition.

### Controlling text size (`box_scale` · `font_size_minimum` · `font_size_minimum_expand_limit` · `font_size_readable_min`)

Four knobs decide how big the translated text is rendered, with distinct, non-overlapping roles:

- **`box_scale`** *(per-task — set in the editor, stored in `pages.json`)* — the main magnifier. It scales each text box **and** the font-size ceiling by the same factor, so `2.5` renders text ~2.5× the detected size and fills the enlarged box (not just adds whitespace). A per-block `scale_exempt` flag opts a single block out.
- **`font_size_minimum`** *(config `[render]`)* — the *desired* lower bound (pixels) for **auto** boxes. Auto boxes now behave like fixed boxes: the font **grows to fill the box** (it is no longer pinned to the detected OCR size); when a long translation won't fit, the box is expanded first (see below) to try to keep the font ≥ this value.
- **`font_size_minimum_expand_limit`** *(config `[render]`)* — expansion fallback. If a long translation still doesn't fit the box_scale'd box even at `font_size_minimum`, the box is expanded further, up to this ratio.
- **`font_size_readable_min`** *(config `[render]`, default `-1` = auto ≈ `(W+H)/300`)* — the **absolute** readability floor, applied to both fixed/user-sized boxes and auto boxes. Fixed boxes never auto-expand, so a long translation shrinks down to this floor; auto boxes, when even the capped expansion can't contain the text, also **shrink the font down to this floor to avoid overflow** (just like fixed boxes) rather than staying pinned at `font_size_minimum` and spilling out. Only text that still won't fit is **flagged in the editor** (see below). Set `4` to restore the old behavior.

Typical workflow: set `box_scale` per task for the overall size you want; leave `font_size_minimum` as the desired floor and `font_size_minimum_expand_limit` as expansion headroom. **Auto and fixed boxes now behave consistently**: both grow the font to fill the box and shrink to fit when needed (auto boxes expand first, then shrink; fixed boxes shrink directly), so auto boxes no longer "get stuck at the minimum size and overflow" while fixed boxes look good. To fine-tune a single box, **resize** it in the editor: the box renders exactly as drawn (never auto-expanded) and the font scales to fill it — drag it bigger and the text grows, smaller and it shrinks (down to `font_size_readable_min`). Merely **moving** a box only repositions it and keeps the automatic sizing. The editor preview uses the same formulas, so what you see matches the output.

### Translator signature (`[signature]`)

Each work can carry a translator credit baked into the output, with a low-key mention of this open-source system folded in. Turn it on in `[signature]`:

```toml
[signature]
enabled    = true
translator = "YourName"          # shown in the credit
pages      = "first_last"        # none | first | last | first_last | every
direction  = "auto"              # auto | horizontal | vertical
position   = "bottom-right"      # which corner
```

The credit renders in a page corner as a **layered mark**: the translator's name is large, drawn on top of a smaller, lighter open-source credit it overlaps. **Both a horizontal and a vertical text box are implemented** — pick one explicitly, or leave `direction = "auto"` for vertical on CJK target languages and horizontal otherwise. By default it appears on the **first and last page** of each work; set `every` for a watermark on every page, or `first` / `last`.

The name line has **no fixed prefix** — the translator decides exactly what to show. It defaults to just the `translator` value and is fully overridable via `text` (placeholder `{translator}`, `\n` for line breaks). The fixed open-source credit — `MTL.downot.moe` — is a separate, auto-lightened layer drawn beneath the name and **cannot be changed or removed**. Name colour, opacity, font size, corner, margin and font are configurable (the credit's lighter shade and smaller size are derived automatically). The signature is **baked at render time and also shown in the editor preview** on the matching pages, so what you see matches the output.

**Resize / reposition in the editor.** Open the region-edit tool (the pencil), then on a page that shows the signature, click it to select it and **drag a corner handle to scale** it or **drag the body to move** it. The size multiplier and offset are saved **per task** in `pages.json` (`signature_scale`, `signature_offset`), so each work can have its own signature size and placement; `render` reads the same values so the baked output matches what you positioned.

### Render QA — automatic problem-block flagging

To make manual review fast at batch scale, the editor automatically flags blocks whose text **overflows** its box (won't fit even at the readability floor) or renders **far smaller** than the rest of the page. Flagged blocks get a red warning badge on their card and a red index tag on the canvas; the **page list shows a red count** of problem blocks per page so you can see at a glance which pages need attention. The **funnel button** in the editor header toggles *show only problem blocks* (with a live count), so you can jump straight to the ones that need a bigger box or a shorter translation. Next to it, a **one-click enlarge** button grows every problem box on the current page just enough to fit its text (kept centred, clamped to the page), with **one-click undo** for that batch. The CLI `render` step logs the same overflow blocks per page.

## Visual Editor (Experimental)

A lightweight web-based visual editor `editor.html` is provided for a better manual review experience.

![Editor Screenshot](screenshot.jpg)
*Example of the Visual Editor (editor.html) in action.*

- **Light / Dark theme**: toggle in the header (remembered across sessions). The canvas uses a neutral grey backdrop so white / black-and-white pages are easy on the eyes.
- **Real-time Preview**: see how the translated text looks on the actual page as you type.
- **Side-by-side original / translation**: each block shows the source text next to an editable translation field for easy comparison.
- **Translation progress at a glance**: per-page `done/total` badges, plus amber highlighting of untranslated blocks on the page list, the canvas tags, and the editor cards.
- **Page thumbnails**: the page list shows lazy-loaded thumbnails (server mode serves cached thumbnails via `api/thumb`).
- **Search & replace**: search across all pages of a task and replace within translations (magnifier button in the editor header).
- **Import / Export translations**: import an external `translations/*.json` and diff it entry-by-entry; export the current language's translations (export shown in server mode).
- **Keyboard Shortcuts**: `←`/`→` page; `↑`/`↓` move between blocks / tasks / pages depending on the focused pane; `Alt`+`←`/`→` page even while editing text; `Alt`/`Ctrl`+`Z` clear the current translation; `?` opens the full shortcut help.
- **Shortcut Focus Routing**: `↑`/`↓` adapt to the active pane — Tasks list, Pages list, dialogue editor (block navigation), or canvas (scroll when zoomed).

### How to Use:
There are two ways to open the editor:
1. **Serverless Local Mode**: Open `editor.html` directly in your browser. Click **Open** to select your `work` folder. (Requires Chrome/Edge, uses the modern HTML5 File System Access API for local reads/writes).
2. **Standalone Server Mode**: Run `python server.py -w ./work`, then open one of the per-task URLs the console prints (each ends with `?t=<token>`). See [Standalone Backend Server](#standalone-backend-server-serverpy).

### Editor tools (regions, pages, merge)

The editor is more than a translation textbox — it can fix layout and geometry directly:

- **Region editor** — click the pencil (region-edit) button in the bottom toolbar to unlock boundary editing; a hint banner stays on screen while it's active. Then:
  - **Resize** a region by dragging its handles, **move** it by dragging the body — the cursor shows which action you'll get. Handles and hit-testing follow the box's **rotation**, so tilted regions (from detection) are just as easy to grab.
  - **Rotate** a region with the round knob above its top edge; the angle snaps near 0/90/180/270° for easy alignment. Rotation is available for every block and is stored per block (`angle`) in `pages.json`.
  - **Draw a new region** by dragging on an empty area (new regions get a white background by default).
  - The **blue box** is the detected text region (where the translation is fitted); the **green dashed box** shows the actual rendered text extent, so you can size the box to control how big the text appears in the bubble. Untranslated blocks are tagged in amber.
  - A hand-adjusted box is marked *fixed* — at render time the text is fitted into exactly that box and is **never auto-expanded**.
- **Render settings popover** — a gear button in the toolbar opens per-task render controls: **box scale**, **min font size**, and **expand limit** (written to `pages.json`; see [Controlling text size](#controlling-text-size-box_scale--font_size_minimum--font_size_minimum_expand_limit)).
- **Per-block background** — each block card has a background toggle that cycles **transparent → match → white**: **transparent** renders text only; **match** fills the block with its estimated region colour (`bg_color`) so the patch blends into a tinted / screentone background instead of a white scar; **white** paints a pure-white rectangle (covers leftover/original content). The match swatch on the button is tinted with the actual fill colour. A delete button removes the block. For a **rotated box**, the background fill now rotates together with the text to match the rendered text area (instead of staying axis-aligned at 0°).
- **Page management** — each page row has an info button (file-path & metadata popup) and a delete button (removes the page and cleans up its blocks, translations in every language, and its clean image; double confirmation).
- **Task merge** — click **Merge**, tick tasks in the order you want them joined (an order badge 1·2·3 appears), then **Confirm**. Pages are renamed sequentially so the merged output keeps one continuous reading order.
- **Import / Export translations** — the import button diffs an external translations file (e.g. `translations/CHS.json`) against the current language entry-by-entry. **Only differences are shown** (identical entries are ignored); tick which to apply (current in red, imported in green) or select all; applied changes are saved automatically. It warns if the file's language differs from the one you're editing. In **server mode** an export button downloads the current language's translations as `<LANG>.json`.

Geometry edits (region/box/render-settings) save to `pages.json` immediately; translations save with the **Save** button (an unsaved-changes dot appears until you do). Re-run `render` to produce the final images.

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
The console prints a secure URL for each task (each ends with `?t=<token>`); open one to edit that task. The server also serves cached page thumbnails (`api/thumb`) and reports pipeline availability to the editor (the Pipeline tab is disabled if `manga_translator_lite` isn't importable on the server).

---

## Translation Review & Story Context Management

For long manga chapters, use the optional **Story-Context-Aware Translation Review and Polish (Review)** command when you intentionally want a separate whole-work pass. It is not part of normal `translate` runs.

### 1. Mechanism & Idempotency
* **Manual trigger**: `python -m manga_translator_lite review ./work` compiles translated blocks in chronological reading order and invokes the LLM using the overall story description. `translate` only creates or refreshes translations.
* **Rolling context**: Translation requests retain recent source → translation pairs only from real preceding pages. Context is cleared at chapter boundaries, and already-approved text on a partially translated page is supplied separately.
* **Strict page-aware prompts**: Prompts include stable block IDs (e.g. `<|p0001_b000|>`) plus optional block semantics. Missing IDs are repaired in a focused follow-up request; no partial response is positionally shifted onto another block.
* **Idempotency**: A `<lang>.reviewed` marker file (e.g. `CHS.reviewed`) is created after a successful manual review. Repeat with `review --overwrite` to force it again.

### 2. Story Context File (`story.txt`)
Simply place a text file named `story.txt` (or `script.txt`, `description.txt`) in your task folder (e.g. `work/task_a/`). The program will automatically search for it. You can also specify it in `pages.json` under the `"story"` key.
Write character bios, relationship details, and style/tone notes in it. The LLM will leverage this outline to perform highly immersive, consistent translations.

If no story file exists, `translate` can generate one before translation by summarizing the OCR dialogue. For multi-chapter tasks, mark chapter starts in the editor; each marker stores `chapter_start` plus an editable `chapter_name` (default `CH1`, `CH2`, ...). User-marked chapters are authoritative and can produce per-chapter story files under `story/<chapter_name>.txt`. If no markers exist, tasks under 30 pages are treated as a single chapter; tasks with 30+ pages may use temporary OCR/title guesses for chapter boundaries, but those guesses are not saved.

### 3. Visual Story Editor & Prompts
Story context management has been fully integrated into the **Visual Editor (`editor.html`)**:
* **New Story Tab**: A sidebar "Story" tab has been added to write, edit, and save `story.txt` on the fly.
* **Chapter Markers**: The page list includes a chapter-start bookmark button. When enabled, it asks for an editable chapter name and saves it to `pages.json`, making later chapter summaries and table-of-contents generation easier.
* **Genre-based Templates**: Built-in multi-lingual (EN/ZH/JA) genre templates (Daily Romance, Fantasy Adventure, Gag/Comedy, Mature/Adult 18+) to bootstrap context writing.
* **Serverless Local Writing**: Utilizing the HTML5 File System Access API, it saves `story.txt` directly to your local workspace disk without needing any active backend server process. It also automatically synchronizes via API when running in `server.py` mode.

### 4. Anti-Censorship Disclaimer
To prevent LLMs (e.g. DeepSeek, Gemini) from rejecting adult or mature manga during translation, a standard English disclaimer ("All characters depicted in the work are entirely fictional and over 18 years old...") is embedded in the system-level prompts, ensuring a smooth translation pipeline.

---

## Cross-language reference translation

When you translate a chapter to several languages, the **first** language usually gets the most human attention — you proofread it, fix names, fix tone. That human judgement (who a pronoun refers to, a character's register, a recurring term) is largely language-independent, so it can give the **next** language a head start. This feature feeds an already-translated language to the LLM as a *reference* while it translates the next one.

* **Source stays the original.** The model still translates from the original text (e.g. Japanese) — the reference is only a semantic / tone hint, never a pivot. The prompt explicitly tells it to disambiguate meaning, referents, names and register from the reference, but **not** to mirror its wording.
* **Read-only & non-destructive.** Each language lives in its own `translations/<LANG>.json`; referencing a language only *reads* it. Translating a new language never touches existing ones, even partially-translated ones. Missing lines in the reference simply fall back to a plain source→target translation.
* **Auto by default.** With no flag, every other language that has been **human-reviewed** (has a `<LANG>.reviewed` marker — see [Translation Review](#translation-review--story-context-management)) is used as reference. Translating the *first* language has nothing to reference, so behaviour is unchanged there.

```bash
# 1) Translate, then hand-correct and optionally review Chinese first
python -m manga_translator_lite translate ./work --target-lang CHS
#    ... proofread CHS in the editor ...
python -m manga_translator_lite review ./work --target-lang CHS

# 2) Translate English — auto-references the reviewed Chinese
python -m manga_translator_lite translate ./work --target-lang ENG

# Reference a specific language only (repeatable); or disable entirely
python -m manga_translator_lite translate ./work --target-lang ENG --reference-lang CHS
python -m manga_translator_lite translate ./work --target-lang ENG --no-reference
```

Resolution (CLI overrides config): `--no-reference` → off; one or more `--reference-lang` → exactly those; neither → config `[translator] reference_langs` (default **auto**). In the **Visual Editor** the Pipeline tab exposes a **Reference Languages** control (Auto / Off / Custom) for the `translate` and `run` commands. Because references improve over time, re-running `translate --overwrite` refreshes only non-hand-edited blocks of the target language using the latest reference, while `edited: true` blocks are preserved.

---

## Translation Spelling & Fluency Proofreading Check

To ensure translations are natural, direct, and free from awkward phrasing or typos, this project includes a **Spelling and Fluency Proofreading Check (Copyediting)** step before rendering.

### 1. Mechanism
* The proofreader sends translated text blocks to the LLM in batches for copyediting.
* It strictly checks for typos, grammatical errors, and awkward phrasing, while completely ignoring punctuation differences to minimize unnecessary updates.
* Changes are presented in a clean table format indicating the original text, current translation, suggestion, and the reason.

### 2. Interaction Modes
When rendering (`render` or `run` command), you can choose how to review recommendations:
* **Interactive Mode (Default on TTY)**: Prompts you to review each recommendation one by one. You can accept (`y`), reject (`n`), manually edit (`e`) the inline text, or quit the review (`q`) and keep the remaining translations as-is.
* **Auto-Apply (`--check -y` or `--check --yes`)**: Automatically accepts and applies all LLM copyediting recommendations without prompting.
* **Force / Bypass**: Use `--check` to force proofreading, or `--no-check` to bypass the proofreading step entirely.

---

## Fixing residual text (`reclean.py`)

Sometimes inpainting leaves bits of the original text around a bubble (faint symbols or handwritten kana that OCR rejected). `reclean.py` re-erases those **without** re-running the whole pipeline — translations and their positions are untouched.

```bash
# Drift-free: re-detect residual text on the clean images and erase it.
# Blocks and translations are NOT modified (no position drift).
python reclean.py work/<task> --redetect

# Then re-render
python -m manga_translator_lite render work/<task> -o out
```

| Option | Description |
|---|---|
| `--redetect` | Re-detect residual text on the clean image and erase it; never touches blocks/translations. Best when your detector config changed since extraction. |
| `--pages 3,7` | Only these pages (1-based, as shown in the editor). |
| `--dilation <px>` | Geometry-mode mask growth (default 35). Ignored with `--redetect`. |
| `--backup` / `--no-backup` | Snapshot each task's `clean/` to a versioned `clean.bak.NNN` before editing (default: on). |
| `--max-backups <n>` | Keep at most N backup versions per task. |

It reads the same `config.toml`, so tuning `[detector]` (e.g. lower `text_threshold`, enable `det_gamma_correct`) improves what it catches.

> `extract` erases regions that pass the primary detector's erase threshold, including symbols/handwriting rejected by translation rules, and records erased non-translation regions as `erase_regions` in `pages.json`. Low OCR confidence no longer makes a primary detector hit miss cleaning; secondary detectors such as RT-DETR derive a local stroke mask by default rather than erasing the whole box. `reclean.py` is for touching up existing tasks.

---

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
cp .env.sample .env  # Configure your API keys
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
