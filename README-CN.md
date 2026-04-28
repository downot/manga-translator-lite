# Manga Image Translator Lite (漫画图像翻译器轻量版)

[English](README.md) | [日本語](README-JP.md) | [中文](README-CN.md)

## 致谢

本项目深感荣幸地感谢 **frederik-uni**、**zyddnys** 及其原始项目 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)。如果没有 [frederik-uni](https://github.com/frederik-uni) 和 [zyddnys](https://github.com/zyddnys) 的杰出工作，本项目将不复存在。此“轻量版”是对原始代码库的简化、模块化和现代化重构。本项目的核心功能仍然基于原始项目。

## 与原项目的核心差异

1.  **解耦的流水线**：不同于原项目的单体或 Web 服务模式，Lite 版将流程拆分为三个独立的 CLI 步骤：`extract` (提取)、`translate` (翻译) 和 `render` (渲染)。中间结果存储在易读的 `pages.json` 中，允许在最终渲染前进行人工审核或编辑。
2.  **LLM 批量优化**：专为大语言模型 (LLM) 重新设计。它支持跨页面的文本块合并，显著降低 API 成本并为翻译提供更好的上下文环境。
3.  **现代化与优化**：完全兼容 Python 3.9+（修复了原项目中的类型提示问题），并针对 Apple Silicon (MPS/Metal 加速) 进行了优化。
4.  **智能渲染**：采用二分搜索算法自动寻找最佳字号，使其在尊重原始检测边界的同时尽量填满气泡区域，从而获得更整洁、更专业的排版效果。
5.  **简化依赖**：去除了沉重的 Web UI 和后端组件，专注于轻量级、CLI 优先的体验，更易于集成到自动化脚本中。

---

本地 OCR + 第三方 LLM API。流水线被拆分为三个可复审的步骤，因此您可以在翻译渲染回页面之前手动对其进行编辑。

```
  in/                                   work/                              out/
  ├── manga_a/                          ├── manga_a/                       ├── manga_a/
  │   ├── 0001.jpg  ── extract ──▶    │   ├── pages.json  ── render ──▶ │   ├── 0001.png
  │   └── 0002.jpg                     │   └── clean/                     │   └── 0002.png
  └── manga_b/                          └── manga_b/                       └── manga_b/
      ├── 0001.jpg                          ├── pages.json                     ├── 0001.png
      └── 0002.jpg                          └── clean/                         └── 0002.png
                                                  ▲
                                                  │ translate (LLM API)
                                                  │ + 手动编辑
```

`in/` 下的每个子目录会被视为一个独立的**任务**。目录结构会被镜像到 `work/` 和 `out/`。
未检测到文字的图片将原样输出——输出图片数量始终与输入保持一致。

## 步骤说明

| 步骤 | 功能 | 输出 |
|---|---|---|
| `extract` | 文本检测 → OCR → 掩码优化 → 图像修复 | `work/<任务名>/clean/*.png`, `work/<任务名>/pages.json` (文本 + 位置) |
| `translate` | 将文本块分组为约1500字符的批次，调用 LLM，填充翻译字段。支持 `--overwrite` 覆盖已有翻译，以及 `--start-index <N>` 从指定页码开始（重新）翻译。 | 各任务更新后的 `pages.json` |
| `render` | 将翻译后的文本绘制到修复后的图像上；无文字页面原样复制 | `out/<任务名>/*.png`（数量与输入一致） |
| `run` | 一键完成 提取 → 翻译 → 渲染 | 两者皆有 |

各任务的 `pages.json` 是唯一的真理来源。在 `translate` 和 `render` 之间打开它以修改任何翻译。

## 快速入门

```bash
pip install -r requirements.txt          # 建议 Python >= 3.10
cp examples/Example.env .env             # 添加 OPENAI_API_KEY 或 GEMINI_API_KEY

python -m manga_translator_lite extract -i ./in -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work -c examples/config-example.toml
python -m manga_translator_lite render ./work -o ./out -c examples/config-example.toml

# 或者全流程运行（跳过手动复审）
python -m manga_translator_lite run -i ./in -w ./work -o ./out -c examples/config-example.toml
```

## 配置说明

支持单个 TOML 或 JSON 文件。所有部分均为可选，默认值已经过优化。

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
target_lang = "CHS"
batch_chars = 1500           # 每个请求约 1000–2000 字符
context_pages = 1            # 发送多少页作为语境参考

[render]
font_size_offset = 0
direction = "auto"
alignment = "auto"
```

`provider = "openai"` 支持任何兼容 OpenAI 的 HTTP 接口，包括 DeepSeek、OpenRouter、Groq 和 Ollama —— 只需将 `api_base` 和 `model` 指向您想要的服务。

API 密钥可以放在 `[translator] api_key` 中，也可以使用环境变量 (`OPENAI_API_KEY` / `GEMINI_API_KEY`)；参见 [examples/Example.env](examples/Example.env)。

使用以下命令打印完整配置结构：

```bash
python -m manga_translator_lite config-help
```

## 可视化编辑器 (实验性)

项目中包含了一个轻量级的网页版可视化编辑器 `editor.html`，用于提供更好的手动校对体验。它可以让你直观地编辑 `pages.json` 数据并实时查看渲染效果。

- **实时预览**：在页面上直接查看译文的最终呈现效果。
- **快速编辑**：在侧边栏修改译文，画布会立即更新。
- **缩放与适应**：支持“最佳比例”和“100% 大小”模式，适配不同尺寸的屏幕。
- **多语言界面**：支持中文、英文和日文界面。
- **快捷键支持**：`←`/`→` 翻页，`Z` 切换缩放，`R` 重新加载，`S` 保存。

### 使用方法：
1. 在项目根目录启动本地服务器（File System Access API 的安全限制需要）：
   ```bash
   python -m http.server 8000
   ```
2. 在现代浏览器（推荐 Chrome 或 Edge）中打开 `http://localhost:8000/editor.html`。
3. 点击 **“打开 work 目录”** 并选择你的 `work` 文件夹即可开始编辑。

## 编辑翻译

在执行 `translate` 之后，各任务目录下的 `pages.json`（例如 `work/manga_a/pages.json`）的结构如下：

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
          "lines": [...],
          "font_size": 24,
          "direction": "auto",
          "alignment": "auto"
        }
      ]
    },
    {
      "index": 1,
      "name": "0002.jpg",
      "size": [1200, 1700],
      "clean": "clean/0001_0002.png",
      "no_text": true,
      "blocks": []
    }
  ]
}
```

编辑任何 `translation` 字段，然后运行 `render`。标记为 `"no_text": true` 的页面没有文本块——它们会被原样复制到输出目录。

### 重新翻译

如果您想从某一页开始重新翻译（例如在修改了前文或调整了配置后），可以使用：

```bash
# 从索引 10 开始重新翻译
python -m manga_translator_lite translate ./work --start-index 10

# 强制重新翻译所有内容
python -m manga_translator_lite translate ./work --overwrite
```


## 项目布局

```
manga_translator_lite/
  pipeline/        提取 / 翻译 / 渲染 核心流程
  detection/       文本检测模块
  ocr/             本地 OCR 模型
  textline_merge/  行合并算法
  mask_refinement/ 掩码优化
  inpainting/      图像修复模块 (AOT, LaMa)
  translators/     统一的 LLM 客户端
  rendering/       文本绘制逻辑
  utils/           共享助手、模型包装、日志
```

## 使用方法

建议使用虚拟环境管理依赖：

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
```

根据以下步骤配置 API 密钥：

```bash
cp examples/Example.env .env

# 修改 .env 文件，填入您的 API 密钥和其他设置
```

在 `in/` 下为每部漫画创建子目录，将图像放入其中，然后运行：

```bash
# in/manga_a/0001.jpg, in/manga_a/0002.jpg, ...
python -m manga_translator_lite extract -i ./in -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## 许可证

GPL-3.0-only。详见 [LICENSE](LICENSE)。
