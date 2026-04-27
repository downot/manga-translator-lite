# Manga Image Translator Lite (漫画画像翻訳器ライト)

## 謝辞

本プロジェクトは、**zyddnys** 氏によるオリジナルプロジェクト [manga-image-translator](https://github.com/zyddnys/manga-image-translator) に深く感謝いたします。zyddnys 氏の卓越した成果がなければ、本プロジェクトは存在しませんでした。この「ライト版」は、オリジナルコードベースを簡素化、モジュール化、および現代化したリファクタリング版です。本プロジェクトのコア機能は、引き続きオリジナル版に基づいています。

## オリジナル版との主な違い

1.  **デカップリングされたパイプライン**: モノリシックなアプローチや Web サービス形式のオリジナルとは異なり、Lite 版ではプロセスを `extract` (抽出)、`translate` (翻訳)、`render` (描画) の 3 つの独立した CLI ステップに分割しました。中間結果は `pages.json` に保存され、最終的な描画の前に手動で確認や編集が可能です。
2.  **LLM バッチ処理の最適化**: 大規模言語モデル (LLM) 向けに設計されています。複数のページにまたがるテキストブロックをバッチ処理することで、API コストを大幅に削減し、より適切な文脈での翻訳を可能にします。
3.  **現代化と最適化**: Python 3.9+ と完全に互換性があり、Apple Silicon (MPS/Metal 加速) 向けに最適化されています。
4.  **スマートレンダリング**: 二分探索アルゴリズムを採用し、元の検出境界を尊重しつつ、吹き出し領域を最大限に埋める最適なフォントサイズを自動的に決定します。
5.  **依存関係の簡素化**: 重い Web UI やバックエンドコンポーネントを削除し、自動化スクリプトに統合しやすい CLI 優先の軽量な構成に焦点を当てました。

---

ローカル OCR + サードパーティ LLM API。パイプラインは 3 つのステップに分かれており、翻訳結果を画像に書き戻す前に手動で編集することができます。

```
  in/                                   work_dir/                          out/
  ┌───────────┐                         ┌─────────────┐                    ┌───────────┐
  │ 0001.jpg  │ ─── extract (CV) ──▶  │ pages.json  │ ── render ──▶     │ 0001.png  │
  │ 0002.jpg  │                       │ clean/0001  │                    │ 0002.png  │
  └───────────┘                         └─────────────┘                    └───────────┘
                                              ▲
                                              │ translate (LLM API)
                                              │ + 手動編集
```

## 手順

| ステップ | 内容 | 出力 |
|---|---|---|
| `extract` | テキスト検出 → OCR → マスク精査 → インペイント | `work_dir/clean/*.png`, `work_dir/pages.json` |
| `translate` | テキストを約1500文字のバッチにまとめ、LLMを呼び出し翻訳を充填 | 更新された `pages.json` |
| `render` | 翻訳されたテキストをインペイント済み画像に描画 | `out_dir/*.png` |
| `run` | 抽出 → 翻訳 → 描画を一括実行 | 両方 |

## クイックスタート

```bash
pip install -r requirements.txt          # Python >= 3.10 推奨
cp examples/Example.env .env             # OPENAI_API_KEY を追加

python -m manga_translator_lite extract -i ./in -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work -c examples/config-example.toml
python -m manga_translator_lite render ./work -o ./out -c examples/config-example.toml
```

## 設定

単一の TOML または JSON ファイルを使用します。すべてのセクションはオプションです。

```toml
use_gpu = false

[detector]
detector = "default"        # default | dbconvnext | ctd | craft | paddle | none
detection_size = 2048

[ocr]
ocr = "48px"                # 32px | 48px | 48px_ctc | mocr

[inpainter]
inpainter = "lama_large"    # default | lama_large | lama_mpe | none

[translator]
provider = "openai"
model = "gpt-4o-mini"
api_base = "https://api.openai.com/v1"
target_lang = "JPN"         # ターゲット言語
batch_chars = 1500           # 1リクエストあたりの文字数
```

## 使用方法

依存関係の管理には仮想環境の使用を推奨します：

```bash
# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

API キーを設定します：

```bash
cp examples/Example.env .env

# .env ファイルを編集して API キーを設定します
```

ルートディレクトリに `in` という名前のフォルダを作成し、漫画の画像を配置して実行します：

```bash
python -m manga_translator_lite extract -i ./in -w ./work -c examples/config-example.toml
python -m manga_translator_lite translate ./work
python -m manga_translator_lite render ./work -o ./out
```

## ライセンス

GPL-3.0-only。詳細は [LICENSE](LICENSE) を参照してください。
