# Manga Image Translator Lite (漫画图像翻译器轻量版)

## 致谢

本项目深感荣幸地感谢 **zyddnys** 及其原始项目 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)。如果没有 zyddnys 的杰出工作，本项目将不复存在。此“轻量版”是对原始代码库的简化、模块化和现代化重构。本项目的核心功能仍然基于原始项目。

## 与原项目的核心差异

1.  **解耦的流水线**：不同于原项目的单体或 Web 服务模式，Lite 版将流程拆分为三个独立的 CLI 步骤：`extract` (提取)、`translate` (翻译) 和 `render` (渲染)。中间结果存储在易读的 `pages.json` 中，允许在最终渲染前进行人工审核或编辑。
2.  **LLM 批量优化**：专为大语言模型 (LLM) 重新设计。它支持跨页面的文本块合并，显著降低 API 成本并为翻译提供更好的上下文环境。
3.  **现代化与优化**：完全兼容 Python 3.9+（修复了原项目中的类型提示问题），并针对 Apple Silicon (MPS/Metal 加速) 进行了优化。
4.  **智能渲染**：采用二分搜索算法自动寻找最佳字号，使其在尊重原始检测边界的同时尽量填满气泡区域，从而获得更整洁、更专业的排版效果。
5.  **简化依赖**：去除了沉重的 Web UI 和后端组件，专注于轻量级、CLI 优先的体验，更易于集成到自动化脚本中。

---

本地 OCR + 第三方 LLM API。流水线被拆分为三个可复审的步骤，因此您可以在翻译渲染回页面之前手动对其进行编辑。

```
  in/                                   work_dir/                          out/
  ┌───────────┐                         ┌─────────────┐                    ┌───────────┐
  │ 0001.jpg  │ ─── extract (CV) ──▶  │ pages.json  │ ── render ──▶     │ 0001.png  │
  │ 0002.jpg  │                       │ clean/0001  │                    │ 0002.png  │
  └───────────┘                         └─────────────┘                    └───────────┘
                                              ▲
                                              │ translate (LLM API)
                                              │ + 手动编辑
```

## 步骤说明

| 步骤 | 功能 | 输出 |
|---|---|---|
| `extract` | 文本检测 → OCR → 掩码优化 → 图像修复 | `work_dir/clean/*.png`, `work_dir/pages.json` (文本 + 位置) |
| `translate` | 将文本块分组为约1500字符的批次，调用配置的 LLM，填充翻译字段 | 更新后的 `pages.json` |
| `render` | 将翻译后的文本绘制到修复后的图像上 | `out_dir/*.png` |
| `run` | 一键完成 提取 → 翻译 → 渲染 | 两者皆有 |

`pages.json` 是唯一的真理来源。在 `translate` 和 `render` 之间打开它以修改任何翻译。

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

在根目录下创建一个名为 `in` 的文件夹，将漫画图像放入其中，然后运行：

```bash
python -m manga_translator_lite extract -i ./in -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## 许可证

GPL-3.0-only。详见 [LICENSE](LICENSE)。
