# Manga Image Translator Lite (漫画图像翻译器轻量版)

[English](README.md) | [日本語](README-JP.md) | [中文](README-CN.md)

## 致谢

本项目深感荣幸地感谢 **frederik-uni**、**zyddnys** 及其原始项目 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)。此“轻量版”是对原始代码库的现代化重构，旨在提供高性能、CLI 优先且具备人工干预灵活性的体验。

## 与原项目的核心差异

1.  **解耦的流水线**：将流程拆分为 `extract` (提取)、`translate` (翻译) 和 `render` (渲染)。中间结果存储在 `pages.json` 中，允许在最终渲染前进行人工审核或编辑。
2.  **LLM 批量优化**：专为大语言模型 (LLM) 设计。支持跨页面的文本块合并，显著降低 API 成本并提供更好的翻译上下文。
3.  **现代化与优化**：完全兼容 Python 3.10+，并针对 Apple Silicon (MPS/Metal) 和 NVIDIA (CUDA) 加速进行了优化。
4.  **智能渲染**：采用二分搜索算法自动寻找最佳字号，在尊重原始检测边界的同时尽量填满气泡区域。
5.  **多任务支持**：自动将 `in/` 下的子目录视为独立的“任务”进行处理，保持清晰的工作区结构。
6.  **增量翻译与恢复**：支持从指定页码恢复任务，并能智能跳过已翻译的内容以节省成本。
7.  **拼写与流利度校对**：在渲染前内置基于大语言模型 (LLM) 的文本审校/校对阶段，用于发现并纠正错别字、语法瑕疵以及生硬或不通顺的译文。
8.  **保留换行排版**：完全支持显式换行符 (`\n`) 对话框格式化，确保浏览器端预览渲染效果与 Python 排版完美一致。

---

本地 OCR + 第三方 LLM API。流水线被拆分为三个可复审的步骤，因此您可以在翻译渲染回页面之前手动对其进行编辑。

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
                                                  │ + 手动编辑 (可视化编辑器)
```

`in/` 下的每个子目录会被视为一个独立的**任务**。目录结构会被镜像到 `work/` 和 `out/`。未检测到文字的图片将原样输出。

## 步骤说明

| 步骤 | 功能 | 输出 |
|---|---|---|
| `extract` | 文本检测 → OCR → 掩码优化 → 图像修复 | `work/<任务>/clean/*.png`, `work/<任务>/pages.json` |
| `translate` | 文本块分组 (~1500 字符)，调用 LLM，填充翻译字段。支持增量更新。 | 各任务更新后的 `pages.json` |
| `render` | 使用智能排版将翻译后的文本绘制到修复后的图像上。可在渲染前选择性地进行拼写与流利度校对（通过 `--check` 选项） | `out/<任务>/*.png` (数量与输入一致) |
| `run` | 一键完成 提取 → 翻译 → 渲染 | 工作目录及最终生成图像 |

各任务的 `pages.json` 是唯一的真理来源。在 `translate` 和 `render` 之间打开它以修改任何翻译。

## 快速入门

```bash
pip install -r requirements.txt          # 建议 Python >= 3.10
cp config.toml.sample config.toml        # 然后填入 [translator] api_key（或改用环境变量）
cp .env.sample .env             # 可选：改用 .env 中的 OPENAI_API_KEY / GEMINI_API_KEY

# 全流程一键运行
python -m manga_translator_lite run -i ./in -w ./work -o ./out

# 或分步骤运行
python -m manga_translator_lite extract -i ./in -w ./work
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## 命令与参数参考

所有命令均通过 `python -m manga_translator_lite <命令> [参数]` 调用。

**通用参数**（`extract`、`translate`、`render`、`run` 均可用）：

| 参数 | 说明 |
|---|---|
| `-c, --config <路径>` | 配置文件（`.toml`/`.json`）路径。缺省时自动使用 `./config.toml`（或 `config.json`）。 |
| `--target-lang <代码>` | 仅对本次运行覆盖 `translator.target_lang`——**不会改动** `config.toml`（如 `CHS`、`ENG`、`JPN`、`KOR`）。 |
| `-v, --verbose` | 详细日志与中间诊断信息。 |

### `extract` — 步骤一：检测 / OCR / 修复 → 工作区

| 参数 | 说明 |
|---|---|
| `-i, --input <路径>` *(必填)* | 输入图片文件，或包含图片 / 子任务文件夹的目录。 |
| `-w, --work-dir <路径>` *(必填)* | 要创建或更新的工作区目录。 |
| `--overwrite` | 即使图片已存在也重新提取；旧译文会按空间 IoU 匹配自动迁移保留。 |

### `translate` — 步骤二：调用 LLM 填充译文

| 参数 | 说明 |
|---|---|
| `work_dir` *(位置参数，必填)* | 已存在的工作区目录。 |
| `--overwrite` | 重新翻译已有译文的块；**人工编辑过的块（`edited: true`）会被保留**。 |
| `--start-index <n>` | 从该页索引开始（重新）翻译；之前的页面仅作为上下文。 |
| `--reference-lang <代码>` | 以某个已翻译语言作为语义/语气参考（可重复指定）。省略即为**自动**（参考所有经人工校对的语言）。见[跨语言参考翻译](#跨语言参考翻译)。 |
| `--no-reference` | 关闭跨语言参考，仅依据原文翻译。 |
| `-j, --concurrency <n>` | 同时并行翻译多少个**任务**（覆盖 `[translator] concurrency`）。**单个任务绝不拆开**，各自保留完整跨页上下文——并发只是同时跑多个独立作品。`1`=串行（默认）。云端 LLM 可设 `3–5` 大幅缩短耗时，注意别超提供方限流；本地/GPU 模型收益不大。 |

> **按语言分文件输出：** `--target-lang` 只临时覆盖配置语言，不会改动 `config.toml`。译文按语言代码命名（`translations/CHS.json`、`translations/ENG.json` ……），因此同一个工作区可以并存多种语言——每种语言翻译一次,互不覆盖：
>
> ```bash
> python -m manga_translator_lite translate ./work --target-lang CHS
> python -m manga_translator_lite translate ./work --target-lang JPN
> ```

### `render` — 步骤三：将译文渲染到清理后的图片

| 参数 | 说明 |
|---|---|
| `work_dir` *(位置参数，必填)* | 已存在的工作区目录。 |
| `-o, --output <路径>` *(必填)* | 最终图片的输出目录。 |
| `--check` | 渲染前强制执行拼写 / 流利度校对。 |
| `--no-check` | 完全跳过校对步骤。 |
| `-y, --yes` | 自动接受并应用所有校对建议。 |

### `run` — 提取 + 翻译 + 渲染 一键完成

接受上述参数的并集：`-i/--input`、`-w/--work-dir`、`-o/--output` *(均必填)*，以及 `--overwrite`、`--check`、`--no-check`、`-y/--yes`，还有翻译阶段的 `--reference-lang`/`--no-reference`。

### `config-help` — 打印配置文件的 JSON Schema

```bash
python -m manga_translator_lite config-help
```

### 编辑器服务端 (`server.py`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-w, --work-dir <路径>` | `work` | 要提供服务的工作区目录。 |
| `-p, --port <n>` | `8000` | 监听端口。 |
| `--host <地址>` | `0.0.0.0` | 绑定地址（设为 `127.0.0.1` 可仅限本机访问）。 |
| `--log-file <路径>` | `server.log` | 服务端日志文件路径。 |

---

## 配置说明

流水线读取单个 TOML（或 JSON）文件。最简单的方式是复制带注释的范例并修改：

```bash
cp config.toml.sample config.toml   # 然后填入 [translator] api_key
```

`config.toml` 已加入 `.gitignore`，密钥不会被提交。完整带注释的参数说明见 **[config.toml.sample](config.toml.sample)**。所有部分均为可选，默认值已经过优化。最小示例：

```toml
use_gpu = true

[detector]
detector = "ctd"            # 选项: ctd | default | dbconvnext | craft | paddle | rtdetr | none
detection_size = -1         # -1=按每页自动计算；也可用 2048/2560 固定全局尺寸
detection_size_scale = 1.0  # 自动模式：小字/密集字漏检时调到 1.3–1.6
detection_size_min = 1024
detection_size_max = 2560
text_threshold = 0.25
box_threshold = 0.6
unclip_ratio = 1.8
secondary_detector = "none" # 设为 "rtdetr" 可融合高召回框检测器

[ocr]
ocr = "48px"                # 选项: 32px | 48px | 48px_ctc | mocr

[translator]
provider = "openai"          # 选项: openai | gemini | none
model = "gpt-4o-mini"
api_base = ""                # 留空用提供商默认值，或填如 https://openrouter.ai/api/v1
api_key = ""                 # 也可留空，改用环境变量 OPENAI_API_KEY
target_lang = "CHS"
batch_chars = 1500           # 每个请求约 1000–2000 字符
context_pages = 1            # 发送前 N 页作为语境参考
temperature = 0.3            # 越低越稳定（人名/语气更一致），越高越发散
use_vision = false           # 为支持视觉的 LLM 附加页面图（更慢/成本更高）
concurrency = 1              # 任务级并发；云端 LLM 常用 3–5
# reference_langs 不设 = 自动（参考所有经人工校对的语言）；[] = 关闭；
# ["CHS"] = 只参考指定语言。被参考的语言全程只读。

[render]
font_path = "fonts/GenEiAntiqueNv5-M.ttf"
font_size_offset = 0
font_size_minimum = 34       # 最小字号下限，保证小字可读
font_size_minimum_expand_limit = 2.5  # 为容纳最小字号，文本框最多允许放大的倍数
font_size_readable_min = -1  # 固定/手画框的自动可读字号下限
line_spacing = 0             # 收紧行距，使同样空间能容纳更大的字
direction = "auto"           # 选项: auto | horizontal | vertical
alignment = "auto"
disable_font_border = false  # 保留文字描边——任何背景上可读性的关键
# font_color = "000000:FFFFFF"  # 黑字+白描边；仅建议用于纯黑白本

[signature]
enabled = false              # 设 true 并填写 translator 即可烘焙署名
translator = ""
pages = "first_last"         # none | first | last | first_last | every
```

`provider = "openai"` 支持任何兼容 OpenAI 的 HTTP 接口（如 DeepSeek、OpenRouter、Groq、Ollama）。API 密钥可放在 `[translator] api_key` 或 `.env`（`OPENAI_API_KEY` / `GEMINI_API_KEY`）中。

范例配置偏保守；混合页面尺寸、追求高召回时，常见实战组合是 `detection_size = -1`、`secondary_detector = "rtdetr"`、`secondary_box_threshold = 0.25–0.3`，云端 LLM 可把 `concurrency` 设为 `3` 左右。共享配置时不要带密钥：把 `api_key = ""` 留空，改用环境变量。

### 选择文字检测器 · RT-DETR

`[detector] detector` 是可插拔的:`default`/`dbconvnext`(DBNet)、`ctd`(Comic Text Detector)、`craft`、`paddle`、`none`,以及实验性的 **`rtdetr`**。

`rtdetr` 封装了 Apache-2.0 的 Hugging Face 模型 `ogkalu/comic-text-and-bubble-detector`(RT-DETR-v2 r50vd),可检测 气泡 / 气泡内文字 / 气泡外文字,在风格化字体、webtoon、manhua 页面上的**区域分型与分组**较强。注意:

- 需要 `transformers`(`pip install transformers`);torch 已是依赖。它是**懒加载**的,不选这个检测器时不影响整包。
- 它是**盒检测器**——返回矩形而非笔画级 mask,擦除 mask 比 DBNet/CTD 粗。生产环境的擦字/重绘请仍用 `ctd`/`default`;目前把 `rtdetr` 当作"检测/区域分型"的实验。
- 置信阈值建议调低:`[detector] box_threshold ≈ 0.3`。
- Apache-2.0 只覆盖代码/权重,**不覆盖**(未披露的)训练数据——商用前请核实来源。

当 RT-DETR 能在风格化 / webtoon / SFX 密集的页面上捕捉到主检测器漏掉的区域时,它最有用。与其整体切换检测器(从而失去笔画级擦除 mask),不如把它**融合**进来——保留产 mask 的检测器(ctd/default)做主检测器,再叠加 RT-DETR 多出的区域。

#### 融合两个检测器（`secondary_detector`）

若第二个检测器能捕捉到主检测器漏掉的区域,你不必二选一——直接**融合**。设置 `[detector] secondary_detector` 后,extract 阶段会同时跑两个检测器:保留主检测器的笔画级 mask 以保证擦除干净,然后把**次检测器检出、而主检测器漏掉的区域**(IoU 低于 `fusion_iou`)加入检测。这些额外区域会照常 OCR、翻译；清图时默认仅提取保守的局部笔画 mask，而不再整框擦除。只有把 `secondary_box_fill = true` 才会恢复旧的整框行为。

```toml
[detector]
detector = "ctd"                # 主检测器——保留笔画检测器以获得干净的 mask
secondary_detector = "rtdetr"   # 增强器——补上 ctd 漏掉的区域(需要 `transformers`)
secondary_box_threshold = 0.3   # rtdetr 偏好比 ctd/dbnet 更低的置信度
fusion_iou = 0.4                # 次检测器区域只有在不与任何主框以高于此 IoU 重叠时才算"新增"
fusion_overlap_limit = 0.5      # 次框盖住/被盖住主框达此比例时也判为重复丢弃（见下）
fusion_max_area_ratio = 0.1     # 丢弃面积超过整页此比例的次检测框（见下）
```

`secondary_detector = "none"`(默认)完全关闭融合——行为不变。这是纯粹的召回率增强:mask 仍由主检测器掌管,因此除了那些额外区域,擦除质量处处与主检测器一致。

**避免重复提取。** 框检测器返回的是大的区域级框,而笔画检测器返回的是小的逐行框 —— 所以一个**盖在**主检测行框**之上**的次检测框,IoU 很低,会被当成「新增」混进来、被 OCR 第二次(常常是残缺的局部副本)。`fusion_overlap_limit` 会在「次框盖住或被盖住任一主框达到该比例(按较小框计交集)」时把它判为重复,在 OCR 之前就丢弃。若同一文字仍被提取两次,调低它(如 `0.3`)。

**避免大面积误擦除。** 框检测器可能把一整块跨越画面的花字标题 / SFX 检成一个大框。`fusion_max_area_ratio` 会丢弃任何面积超过整页该比例的次检测框(默认 `0.1` = 10%),这样超大区域框既不翻译也不擦除,画面保持原样。若仍有不确定的大框，调低它(如 `0.06`);设为 `0` 则关闭该上限。

#### 显存与速度

每页的检测与修复(inpaint)会在不同尺寸的大张量间交替(如 `detection_size` 2560 后接 `inpainting_size` 2048),这会让 CUDA 显存碎片化。项目内置了几项措施，让"高召回设置"(大 `detection_size` **加** 次检测器)也能控制在预算内：

- **RT-DETR 在 CUDA 上自动以 fp16 运行**——日志会显示 `RT-DETR running in fp16 (CUDA)`——次检测器显存约减半，且召回不变(CPU/MPS 保持 fp32)。
- **每页在检测阶段与修复阶段之间释放显存**，两个阶段的峰值分配不再叠加、也不跨页碎片化。
- **默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**(在包初始化、torch 初始化分配器之前),用于抑制碎片化。你可以在环境里 export 自己的值来覆盖；在不支持该选项的 Windows / 旧版 torch 上，torch 会直接忽略它。

调节旋钮（从最安全到最影响质量）：

- **`det_gamma_correct`**——普通(亮底高对比)漫画请**关闭**。它会把亮底页自动压暗成中灰，反而**降低**检测/擦除质量并增加每页开销；只有真正偏暗/发灰的扫图才该开。
- **`inpainting_size`**(如 2048 → 1536)——压低修复阶段显存，对擦除后的文字几乎无可见影响；是"显存换质量"性价比最高的旋钮。
- **`detection_size`** 与 **`secondary_detector`**——你最强的召回杠杆，所以**最后**才动它们：缩小 `detection_size` 或关掉次检测器能省显存，但识别率会明显下降。

### 控制文字大小（`box_scale` · `font_size_minimum` · `font_size_minimum_expand_limit` · `font_size_readable_min`）

四个旋钮共同决定译文渲染多大，各司其职、互不重叠：

- **`box_scale`**（任务级——在编辑器里设置，存入 `pages.json`）——主放大系数。它把每个文本框**和字号上限**按同一比例放大，所以 `2.5` 会让文字约为检测尺寸的 2.5×、填满放大后的框（而不是只增加留白）。某个块可用 `scale_exempt` 单独豁免。
- **`font_size_minimum`**（config `[render]`）——**自动框**的理想字号下限（像素）。自动框现在和固定框一样，字号会**长大到填满文本框**（不再被钉死在原文检测出的字号上）；当长译文塞不下时，优先把框扩大（见下）以尽量保持 ≥ 此值。
- **`font_size_minimum_expand_limit`**（config `[render]`）——兜底扩框。若长译文在 box_scale 后的框里、缩到 `font_size_minimum` 仍塞不下，才把框进一步扩大，最多到这个倍数。
- **`font_size_readable_min`**（config `[render]`，默认 `-1`=自动 ≈ `(宽+高)/300`）——可读性的**绝对下限**，对固定/手画框和自动框都生效。固定/手画框不自动扩框，长译文会一路缩字到此为止；自动框在扩框到上限仍放不下时，也会**兜底缩字**到此值以**保证不溢出**（与固定框一致），而不是卡在 `font_size_minimum` 硬撑导致文字溢出。仍塞不下才在编辑器里**标记该块**（见下）。设 `4` 可恢复旧行为。

典型用法：按任务设 `box_scale` 得到整体想要的大小；`font_size_minimum` 留作理想下限、`font_size_minimum_expand_limit` 留作扩框兜底。**自动框与固定框现在表现一致**：字号都会长大到填满框，放不下时都会缩字贴合（自动框先尝试扩框、再缩字，固定框直接缩字），所以自动框不会再出现"卡在最小字号上溢出"而固定框却好看的落差。要精修单个框就在编辑器里**缩放**它：该框严格按所画渲染（不自动扩框），字号自适应框大小——拉大框文字随之变大、缩小框文字随之变小以填满框（下限为 `font_size_readable_min`）。仅仅**移动**框只是重定位，保留原来的自动字号。编辑器预览与渲染使用同一套公式，所见即所得。

### 译者署名（`[signature]`）

每部作品都可以把译者署名烘焙进成品页，并把这套开源系统以低调形式并入署名。在 `[signature]` 里开启：

```toml
[signature]
enabled    = true
translator = "你的名字"           # 显示在署名里
pages      = "first_last"        # none | first | last | first_last | every
direction  = "auto"              # auto | horizontal | vertical
position   = "bottom-right"      # 放在哪个角落
```

署名在页面角落渲染成**分层叠压**效果：译者名较大、压在上面，下方是更小更浅的开源标注、两者相叠。**横向和纵向两套文本框都已实现**——可显式选一种，或用 `direction = "auto"`：CJK 目标语言自动纵向、其他横向。默认出现在每部作品的**首尾两页**；设 `every` 可每页水印，或 `first` / `last`。

译者名那行**没有固定前缀**——由译者自己决定展示什么，默认就是 `translator` 的值，可用 `text` 完全自定义（占位符 `{translator}`，`\n` 换行）。固定开源标注 `MTL.downot.moe` 是绘制在译者名下方、**自动调浅**的独立一层，**不可修改或移除**。译者名的颜色、不透明度、字号、角落、边距、字体均可配置（标注那层的更浅颜色与更小字号会自动推导）。署名**在 render 时烘焙进图片，同时在编辑器预览的对应页叠加显示**，所见即所得。

**在编辑器里缩放 / 移动**：打开区域编辑工具（铅笔），在显示署名的页面上点击署名选中它，然后**拖角上的手柄缩放**、或**拖动主体移动**。缩放倍数与偏移按**任务**保存到 `pages.json`（`signature_scale`、`signature_offset`），所以每部作品可以有各自的署名大小与位置；`render` 读取同样的值，烘焙输出与你摆放的一致。

### 渲染质检——自动标记问题块

为了让大批量人工复核更快，编辑器会自动标记两类问题块：文字**溢出**文本框（即便到了可读性下限仍塞不下），或渲染得**明显小于**本页其他块。被标记的块在卡片上显示红色警告徽章、在画布上显示红色序号标签；**页面列表会显示每页问题块的红色计数**，一眼就能看出哪几页需要处理。编辑器头部的**漏斗按钮**可切换“只看问题块”（带实时计数），直接跳到需要放大框或精简译文的那些块。漏斗旁还有一个**一键放大**按钮：对当前页的所有问题块，按文字实际需要把框放大到刚好放下（中心不动、自动避免越界），并支持**一键撤销**本次放大。CLI 的 `render` 步骤也会按页打印这些溢出块。

## 可视化编辑器 (实验性)

项目包含了一个轻量级的网页版可视化编辑器 `editor.html`，用于提供更好的手动校正体验。

![编辑器截图](screenshot.jpg)
*可视化编辑器 (editor.html) 的实际运行效果示例。*

- **浅色 / 深色主题**：顶部一键切换（记忆到下次打开）。画布使用中性灰底，让白底 / 黑白漫画页更护眼。
- **实时预览**：边输入边在页面上查看译文最终效果。
- **原文 / 译文并排**：每个文字块左右并排显示原文与可编辑译文框，便于逐句对照。
- **翻译进度一目了然**：页面列表显示 `已译/总数` 角标；未译文字块在列表、画布序号标签与编辑卡片上以琥珀色高亮。
- **页面缩略图**：页面列表显示懒加载缩略图（服务端模式经 `api/thumb` 提供带缓存的缩略图）。
- **跨页搜索 / 替换**：在整个任务范围内搜索并替换译文（编辑器标题栏的放大镜按钮）。
- **导入 / 导出译文**：导入外部 `translations/*.json` 并逐条对比差异；导出当前语言译文（导出按钮在服务端模式下显示）。
- **快捷键**：`←`/`→` 翻页；`↑`/`↓` 按当前焦点区域在 文字块 / 任务 / 页面 间移动；`Alt`+`←`/`→` 在编辑文本时也能翻页；`Alt`/`Ctrl`+`Z` 清空当前译文；`?` 打开完整快捷键帮助。
- **快捷键智能焦点路由**：`↑`/`↓` 随当前区域自适应 —— 任务列表、页面列表、对话框编辑（切换文字块）、画布（放大时滚动）。

### 使用方法：
支持以下两种方式打开编辑器：
1. **纯本地离线模式**：直接在浏览器中双击打开 `editor.html`，点击 **“打开”** 并选择您的 `work` 文件夹。（基于现代 HTML5 File System Access API，无需后端运行即可直接安全读写本地磁盘）。
2. **联机后端模式**：运行 `python server.py -w ./work`，然后打开终端打印的任务链接（每个以 `?t=<token>` 结尾）。详见下文「独立后端 API 服务」。

### 编辑器工具（区域 / 页面 / 合并）

编辑器不只是译文输入框，还能直接修正排版与几何：

- **区域编辑**：点底部工具栏的铅笔（区域编辑）按钮解锁边界编辑，开启时画布上常驻提示条。然后：
  - 拖**手柄**缩放区域、拖**框体**移动（鼠标光标会提示当前是拉伸还是移动）；手柄与命中检测会**跟随框的旋转角度**，所以检测得到的倾斜框也能轻松抓取；
  - 用框顶边上方的**圆形旋钮旋转**区域，角度在 0/90/180/270° 附近自动吸附以便对齐；所有块都可旋转，角度按块存入 `pages.json`（`angle`）；
  - 在**空白处拖拽**新建区域（新区域默认带白色背景）；
  - **蓝框** = 检测到的文字区域（译文塞入的目标框）；**绿色虚线** = 译文实际渲染范围，据此调整蓝框即可控制文字在气泡里的大小；未翻译的块序号标签为琥珀色；
  - 手动调过的框标记为 *fixed*，渲染时文字严格塞进该框、**不再自动扩大**。
- **渲染设置浮层**：工具栏的齿轮按钮打开本任务的渲染参数：**文本框缩放**、**最小字号**、**扩框上限**（写入 `pages.json`，详见下文“控制文字大小”）。
- **逐块背景**：每个块卡片有背景切换，循环 **透明 → 同色 → 白底**：**透明**只渲染文字；**同色**用该块估计的区域底色（`bg_color`）填充，让覆盖块融入彩底/网点背景，而不是留下一块白疤；**白底**铺纯白矩形（盖住残留/原内容）。按钮上的色块会染成实际填充色。删除按钮可删除该块。**带旋转角度的框**，其背景填充会随文字一起旋转、贴合渲染后的文字区域（不再固定为 0° 水平）。
- **页面管理**：每个页面行有信息按钮（文件路径与元数据弹窗）和删除按钮（删除该页并清理其区块、所有语言译文、clean 图；二次确认）。
- **任务合并**：点 **合并**，按想要的拼接顺序勾选任务（出现序号徽章 1·2·3），再 **确认**。页面会被顺序重命名，使合并输出保持单一连续的阅读顺序。
- **导入 / 导出译文**：导入按钮将外部译文文件（如 `translations/CHS.json`）按条与当前语言对比，**只显示差异**（相同忽略）；勾选要应用的项（当前=红、导入=绿）或全选，应用后自动保存；语言不一致时会提示。**服务端模式**下还有导出按钮，可把当前语言译文下载为 `<LANG>.json`。

几何改动（区域/框/渲染设置）会即时写入 `pages.json`；译文用 **保存** 按钮保存（未保存时按钮上有提示圆点）。完成后重跑 `render` 生成最终图。

---

## 独立后端 API 服务 (`server.py`)

除了简单的静态文件托管，本项目现在内置了一个功能完备的轻量级 Python 后端服务（`server.py`），用于管理任务数据、自动同步剧本、并直接在浏览器内触发管道流程。

### 1. 核心功能
* **网页端一键执行管道**：直接在网页编辑器中一键运行 `Extract` (提取)、`Translate` (翻译)、`Render` (渲染) 或完整的串联 `Run` 流程，日志将通过 Server-Sent Events (SSE) 实时推送到浏览器终端窗口。
* **数据自动同步与存盘**：支持将页面数据、手动修改的译文以及剧本内容 (`story.txt`) 直接高速存盘，规避浏览器的沙箱安全限制。
* **基于 Token 的安全隔离**：为每个任务目录基于工作区特征自动生成专属访问 Token，防止未经授权的误操作。

### 2. 使用方法
在您的工作目录启动服务端：
```bash
python server.py -w ./work -p 8000
```
终端会为每个任务打印一个安全链接（以 `?t=<token>` 结尾），打开即可编辑该任务。服务端还会提供带缓存的页面缩略图（`api/thumb`），并向编辑器报告 Pipeline 是否可用（若服务端无法 import `manga_translator_lite`，Pipeline 标签会被禁用）。

---

## 译文自动润色与剧本管理

为解决长篇或连贯剧情漫画在分批翻译时可能出现的语气不连贯、上下文脱节等问题，本项目在 `translate` 阶段引入了**故事剧本级译文自动润色 (Review)** 功能。

### 1. 运行原理与幂等性
* **润色时机**：在 `translate` 步骤完成初版翻译后，系统会自动组织整章对话，根据**整体故事剧本描述**对所有译文进行全局连贯性与角色语气润色。
* **滚动上下文**：翻译批次之间会继续传递最近的「原文 → 译文」对照，让人名、代词、敬语、角色语气在长任务里更稳定。
* **按页组织 Prompt**：初译 prompt 会插入页分隔，并使用稳定的文字块 ID（如 `<|p0001_b000|>`）对齐结果，降低错位概率。整批失败或漏项时会降级为单块重试，最终仍为空的块不会覆盖已有译文。
* **幂等性保障**：润色成功后会在 `translations` 下生成 `<目标语言>.reviewed` 标记文件。再次执行 `translate` 会自动跳过已润色的任务。
* **增量与补齐**：对于已翻译但尚未润色的任务，再次执行 `translate` 将直接跳过翻译阶段，**智能补齐润色流程**；使用 `--overwrite` 参数则会强制重新翻译并重新润色。

### 2. 故事剧本文件 (`story.txt`)
您只需在任务工作目录下（如 `work/task_a/`）放置一个描述文件，程序会自动按顺序检索 `story.txt`/`script.txt`/`description.txt` 等。您也可以直接在 `pages.json` 中配置 `"story"` 字段。
在文件中写入该漫画的故事背景、角色性格、人设关系和语气偏好，大模型将以此为参考执行极具沉浸感的高级汉化润色。

如果没有故事文件，`translate` 会在初译前尝试根据 OCR 全文对白自动总结并生成故事上下文。对于多章节任务，建议在编辑器中标记新章节：每个标记会保存 `chapter_start` 和可编辑的 `chapter_name`（默认 `CH1`、`CH2`……）。人工章节标记优先，并可生成 `story/<章节名>.txt` 这类分章节剧本。若没有人工标记，少于 30 页的任务默认视为单章节；30 页及以上才会根据 OCR/文件名临时猜测章节边界，且这些猜测不会写回文件。

### 3. 可视化剧本编辑器与模版提示器
我们已将剧本管理全面整合进**可视化编辑器 (`editor.html`)**：
* **全新 Story 面板**：在侧边栏新增了 "Story" 标签页，支持直接编辑并保存 `story.txt`。
* **章节标记**：页面列表新增章节开始书签按钮。开启时会要求输入章节名（可编辑），并保存进 `pages.json`，方便后续生成分章节摘要和目录。
* **多题材模版**：内置多语种（中/英/日）漫画类型模版（都市日常恋爱、异世界奇幻冒险、搞笑脑洞喜剧、青年成人内容 18+），支持一键套用并快捷修改。
* **无依赖离线存盘**：采用现代 HTML5 File System Access API，无需运行 `server.py` 即可在本地浏览器中直接安全地将剧本写入您磁盘上的任务目录。在 `server.py` 联机模式下亦可自动通过 API 同步。

### 4. 防止内容审查 (过审标识)
为防范成人内容漫画在翻译或润色过程中遭遇大语言模型（如 DeepSeek、Gemini）的安全防护与敏感词风控拦截，本项目在底层的系统级 Prompt 中植入了天然的免责申明（所有人物纯属虚构且均已成年 18+），保障翻译流程顺畅无阻。

---

## 跨语言参考翻译

当一章要翻译成多种语言时，**第一种**语言往往投入了最多人工——你会校对它、修人名、调语气。这些人工判断（代词指代谁、角色的语气、反复出现的术语）大多与语言无关，因此可以给**下一种**语言开一个好头。该功能在翻译下一种语言时，把某个已翻译语言作为**参考**喂给 LLM。

* **原文始终是源。** 模型仍然从原文（如日文）翻译——参考只是语义/语气提示，绝非二次跳板翻译。Prompt 明确要求它据参考消歧含义、指代、专名与语域，但**不要**照抄其措辞。
* **只读、不破坏。** 每种语言独立存放于 `translations/<语言>.json`，参考某语言只会**读取**它。翻译新语言绝不触碰已有语言，即便它们尚未译完；参考中缺失的行会自动退回普通的原文→目标翻译。
* **默认自动。** 不带任何参数时，所有经过**人工校对**（含 `<语言>.reviewed` 标记——见[译文自动润色](#译文自动润色与剧本管理)）的其它语言都会被用作参考。翻译*第一种*语言时无可参考，行为与以往一致。

```bash
# 1) 先翻译 + 润色 + 人工校正中文
python -m manga_translator_lite translate ./work --target-lang CHS
#    ... 在编辑器里校对中文 ...

# 2) 翻译英文——自动参考已校对的中文
python -m manga_translator_lite translate ./work --target-lang ENG

# 只参考指定语言（可重复）；或彻底关闭
python -m manga_translator_lite translate ./work --target-lang ENG --reference-lang CHS
python -m manga_translator_lite translate ./work --target-lang ENG --no-reference
```

解析优先级（CLI 覆盖配置）：`--no-reference` → 关闭；一个或多个 `--reference-lang` → 仅参考这些；都不给 → 取配置 `[translator] reference_langs`（默认**自动**）。在**可视化编辑器**中，Pipeline 面板为 `translate` 和 `run` 命令提供了**参考语言**控件（自动 / 关闭 / 自定义）。由于参考会随时间变好，重新运行 `translate --overwrite` 会用最新参考刷新目标语言中**非人工编辑**的块，而 `edited: true` 的块会被保留。

---

## 译文拼写与流利度校对

为确保翻译结果自然、地道，且没有拼写错误或生硬表达，本项目在渲染前引入了**译文拼写与流利度校对（审校）**步骤。

### 1. 运行机制
* 校对模块会将翻译后的文本块分批发送给大语言模型进行审校。
* 它会严格检查错别字、语法瑕疵以及不流利的表达，同时**完全忽略**标点符号的差异，以尽量减少不必要的更新。
* 发现的更改建议将以清晰的列表格式展示，指出原文、当前翻译、建议修改以及修改原因。

### 2. 交互模式
在渲染（`render` 或 `run` 命令）时，您可以选择以下审核方式：
* **交互模式（TTY/终端环境下的默认行为）**：程序会逐条提示您确认每项建议，您可以接受（`y`）、拒绝（`n`）、手动修改（`e`）或退出审核（`q`）。
* **自动应用 (`--check -y` 或 `--check --yes`)**：直接自动接受并应用大语言模型的所有校对建议，无需手动确认。
* **强制/绕过**：使用 `--check` 选项强制启动校对步骤，或者使用 `--no-check` 选项完全绕过它。

---

## 修复残留原文（`reclean.py`）

有时修复（inpaint）后气泡周围会残留一点原文（OCR 判定为非文字的符号或手写假名）。`reclean.py` 可以**在不重跑整条流水线**的前提下重新擦除它们——译文及其位置完全不动。

```bash
# 无漂移：在 clean 图上重新检测残留并擦除，不改 blocks/译文（位置零漂移）
python reclean.py work/<任务> --redetect

# 之后重新渲染
python -m manga_translator_lite render work/<任务> -o out
```

| 参数 | 说明 |
|---|---|
| `--redetect` | 在 clean 图上重新检测残留并擦除；不碰 blocks/译文。**配置变过后最适用** |
| `--pages 3,7` | 仅这些页（1-based，对应编辑器里的页码） |
| `--dilation <px>` | 几何模式的 mask 膨胀（默认 35）；`--redetect` 下忽略 |
| `--backup` / `--no-backup` | 改动前把任务的 `clean/` 快照成多版本 `clean.bak.NNN`（默认开） |
| `--max-backups <n>` | 每个任务最多保留 N 个备份版本 |

它读取同一份 `config.toml`，所以调 `[detector]`（如降低 `text_threshold`、开启 `det_gamma_correct`）能提升擦除命中。

> `extract` 会擦除通过主检测器擦除阈值的区域（含被翻译规则拒掉的符号/手写），并把未进入翻译块的已擦区域记为 `pages.json` 的 `erase_regions`。OCR 低置信不会再让主检测命中漏擦；次检测器（如 RT-DETR）默认只提取局部笔画 mask，避免整框误擦。`reclean.py` 用于修补存量任务。

---

## 编辑翻译

执行 `translate` 之后，各任务目录下的 `pages.json` 结构如下：

```json
{
  "version": 2,
  "target_lang": "CHS",
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
          "translation": "早上好",
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

### 重新翻译与恢复

Lite 支持智能增量更新与断点续传：

```bash
# 从索引 10 开始重新翻译
python -m manga_translator_lite translate ./work --start-index 10

# 强制重新翻译所有内容（覆盖已有译文）
python -m manga_translator_lite translate ./work --overwrite
```

## 项目布局

```text
manga_translator_lite/
  pipeline/        # 核心 CLI 步骤 (extract, translate, render, run)
  translators/     # 统一的 LLM 客户端 (兼容 OpenAI, Gemini)
  rendering/       # 智能排版与字体适配
  detection/       # 文本检测模块
  ocr/             # 本地 OCR 封装
  ...
```

## 使用方法

建议使用虚拟环境管理依赖：

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.sample .env  # 配置 API 密钥
```

## 许可证

GPL-3.0-only。详见 [LICENSE](LICENSE)。
