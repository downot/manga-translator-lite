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
| `render` | 使用智能排版将翻译后的文本绘制到修复后的图像上 | `out/<任务>/*.png` (数量与输入一致) |
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
direction = "auto"           # 选项: auto | horizontal | vertical
alignment = "auto"
```

`provider = "openai"` 支持任何兼容 OpenAI 的 HTTP 接口（如 DeepSeek, Groq, Ollama）。API 密钥可放在 `[translator] api_key` 或 `.env` 中。

## 可视化编辑器 (实验性)

项目包含了一个轻量级的网页版可视化编辑器 `editor.html`，用于提供更好的手动校正体验。

- **实时预览**：在页面上直接查看译文的最终呈现效果。
- **快速编辑**：在侧边栏修改译文，画布会立即更新。
- **快捷键**：`←`/`→` 翻页，`Z` 切换缩放，`R` 重新加载，`S` 保存。

### 使用方法：
1. 启动本地服务：`python -m http.server 8000`
2. 在浏览器（推荐 Chrome/Edge）中打开 `http://localhost:8000/editor.html`。
3. 点击 **“打开工作目录”** 并选择你的 `work` 文件夹。

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
