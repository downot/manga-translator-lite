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
cp examples/Example.env .env             # 添加 OPENAI_API_KEY 或 GEMINI_API_KEY

# 全流程一键运行
python -m manga_translator_lite run -i ./in -w ./work -o ./out

# 或分步骤运行
python -m manga_translator_lite extract -i ./in -w ./work
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## 配置说明

支持单个 TOML 或 JSON 文件。所有部分均为可选，默认值已经过优化。

```toml
use_gpu = true

[detector]
detector = "default"        # 选项: default | dbconvnext | ctd | craft | paddle
detection_size = 2048

[ocr]
ocr = "48px"                # 选项: 32px | 48px | 48px_ctc | mocr

[translator]
provider = "openai"          # 选项: openai | gemini
model = "gpt-4o-mini"
api_base = "https://api.openai.com/v1"
target_lang = "CHS"
batch_chars = 1500           # 每个请求约 1000–2000 字符
context_pages = 2            # 发送前 N 页作为语境参考

[render]
font_size_offset = 0
font_size_minimum = 18       # 允许的最小字体大小，防止文本过小
font_size_minimum_expand_limit = 1.5  # 当文字不适配时，允许放大文本框的最大比例
direction = "auto"           # 选项: auto | horizontal | vertical
alignment = "auto"
```

`provider = "openai"` 支持任何兼容 OpenAI 的 HTTP 接口（如 DeepSeek, Groq, Ollama）。API 密钥可放在 `[translator] api_key` 或 `.env` 中。

## 可视化编辑器 (实验性)

项目包含了一个轻量级的网页版可视化编辑器 `editor.html`，用于提供更好的手动校正体验。

![编辑器截图](screenshot.jpg)
*可视化编辑器 (editor.html) 的实际运行效果示例。*

- **实时预览**：在页面上直接查看译文的最终呈现效果。
- **快速编辑**：在侧边栏修改译文，画布会立即更新。
- **快捷键**：`←`/`→` 翻页，`Z` 切换缩放，`R` 重新加载，`S` 保存。
- **快捷键智能焦点路由**：根据用户当前操作的区域，智能自适应 `↑`/`↓` 方向键功能：
  - **任务列表区域 (Tasks)**：进行不同漫画任务的上下快速切换。
  - **页面列表区域 (Pages)**：进行页面索引的上下快速切换。
  - **对话框编辑区域 (Editor)**：在编辑文本框时，上下键切换不同的对话文本块。
  - **画布区域 (Canvas)**：在图片放大状态下，上下键可以顺畅地上下滚动图片。
- **任务列表独立滚动**：优化了左侧任务列表的布局，支持超长任务列表独立滚动，防止列表过长撑破页面布局。

### 使用方法：
支持以下两种方式打开编辑器：
1. **纯本地离线模式**：直接在浏览器中双击打开 `editor.html`，点击 **“打开工作目录”** 并选择您的 `work` 文件夹。（基于现代 HTML5 File System Access API，无需后端运行即可直接安全读写本地磁盘）。
2. **联机后端模式**：运行 `python server.py -w ./work` 后，在浏览器中访问 `http://localhost:8000/editor.html`。

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
启动后在浏览器中访问 `http://localhost:8000/editor.html`，终端会打印出每个 Task 对应的安全访问链接。

---

## 译文自动润色与剧本管理

为解决长篇或连贯剧情漫画在分批翻译时可能出现的语气不连贯、上下文脱节等问题，本项目在 `translate` 阶段引入了**故事剧本级译文自动润色 (Review)** 功能。

### 1. 运行原理与幂等性
* **润色时机**：在 `translate` 步骤完成初版翻译后，系统会自动组织整章对话，根据**整体故事剧本描述**对所有译文进行全局连贯性与角色语气润色。
* **幂等性保障**：润色成功后会在 `translations` 下生成 `<目标语言>.reviewed` 标记文件。再次执行 `translate` 会自动跳过已润色的任务。
* **增量与补齐**：对于已翻译但尚未润色的任务，再次执行 `translate` 将直接跳过翻译阶段，**智能补齐润色流程**；使用 `--overwrite` 参数则会强制重新翻译并重新润色。

### 2. 故事剧本文件 (`story.txt`)
您只需在任务工作目录下（如 `work/task_a/`）放置一个描述文件，程序会自动按顺序检索 `story.txt`/`script.txt`/`description.txt` 等。您也可以直接在 `pages.json` 中配置 `"story"` 字段。
在文件中写入该漫画的故事背景、角色性格、人设关系和语气偏好，大模型将以此为参考执行极具沉浸感的高级汉化润色。

### 3. 可视化剧本编辑器与模版提示器
我们已将剧本管理全面整合进**可视化编辑器 (`editor.html`)**：
* **全新 Story 面板**：在侧边栏新增了 "Story" 标签页，支持直接编辑并保存 `story.txt`。
* **多题材模版**：内置多语种（中/英/日）漫画类型模版（都市日常恋爱、异世界奇幻冒险、搞笑脑洞喜剧、青年成人内容 18+），支持一键套用并快捷修改。
* **无依赖离线存盘**：采用现代 HTML5 File System Access API，无需运行 `server.py` 即可在本地浏览器中直接安全地将剧本写入您磁盘上的任务目录。在 `server.py` 联机模式下亦可自动通过 API 同步。

### 4. 防止内容审查 (过审标识)
为防范成人内容漫画在翻译或润色过程中遭遇大语言模型（如 DeepSeek、Gemini）的安全防护与敏感词风控拦截，本项目在底层的系统级 Prompt 中植入了天然的免责申明（所有人物纯属虚构且均已成年 18+），保障翻译流程顺畅无阻。

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
cp examples/Example.env .env  # 配置 API 密钥
```

## 许可证

GPL-3.0-only。详见 [LICENSE](LICENSE)。
