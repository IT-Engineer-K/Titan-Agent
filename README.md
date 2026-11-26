# TitanAgent

Titans（*Learning to Memorize at Test Time*, arXiv:2501.00663）の長期記憶モジュール（MAC: Memory as a Context）をベースにした、1本道の超長期会話で育つパーソナル AI エージェントの実験リポジトリです。

全体の実験方針は [PLAN.md](PLAN.md) を参照してください。

## 実験1: 日本語特化 Titans (MAC) のゼロからの事前学習

- ノートブック: `exp1_pretrain_ja/exp1_titans_ja_pretrain.ipynb`
- CLI スクリプト: `exp1_pretrain_ja/train_titans_ja.py`

### Google Colab（有料枠）での実行手順
1. [Google Colab](https://colab.research.google.com/) を開き、`exp1_pretrain_ja/exp1_titans_ja_pretrain.ipynb` をアップロードします。
2. ランタイムのタイプを **GPU（L4 / A100 推奨）** に変更します。
3. まずは `PRESET = 'tiny'`、`MAX_STEPS = 1000` のまま上から順に実行して動作確認し、問題なければ `small` や `medium` に上げて学習します。

### Kaggle（無料枠・バックグラウンド最大12時間）での実行手順
1. [Kaggle Notebooks](https://www.kaggle.com/code) で新規ノートブックを作成し、`exp1_pretrain_ja/exp1_titans_ja_pretrain.ipynb` をインポートします。
2. 右パネルの Settings で **Accelerator = GPU T4 x2（または P100）**、**Internet = On** に設定します。
3. **Save Version → Save & Run All (Commit)** でバックグラウンド実行を開始すると、ブラウザを閉じても最大12時間学習が継続します。

## 実験1.5: 小型 Titans (MAC) の段階的カリキュラム学習（SFT → GRPO）

「検索 → 概念の連鎖整理 → 記憶の再利用」の3段階ミニ勉強ループを、Kaggle 無料枠（T4/P100 16GB）で OOM を避けながら検証する構成です。

- ノートブック: `exp1_curriculum_rl/exp1_titans_curriculum_rl.ipynb`
- CLI スクリプト: `exp1_curriculum_rl/train_titans_curriculum.py`

### Kaggle での実行手順
1. [Kaggle Notebooks](https://www.kaggle.com/code) で新規ノートブックを作成し、`exp1_curriculum_rl/exp1_titans_curriculum_rl.ipynb` をインポートします。
2. Settings で **Accelerator = GPU T4 x2（または P100）**、**Internet = On** に設定します。
3. デフォルト（`PRESET = 'tiny'`, `SFT_STEPS = 800`, `GRPO_STEPS = 120`, `GROUP_SIZE = 4`）のまま上から順に実行すると、Stage 1（SFT）→ 評価 → Stage 2（GRPO）→ 比較評価まで一括で回ります。

## 実験2: 小型 Titans (MAC) による思考整理・壁打ち学習（SFT → GRPO）

「不足前提を見抜く逆質問（Level 1）→ 複数前提の多段深掘り（Level 2）→ 過去の前提を聞き直さない長期記憶の壁打ち（Level 3）」を検証する構成です。実験1.5の知見を踏まえ、段階報酬・Level 2/3 重点 GRPO・各レベル40問評価を組み込んでいます。

- ノートブック: `exp2_thinking_partner/exp2_titans_thinking_partner.ipynb`
- CLI スクリプト: `exp2_thinking_partner/train_thinking_partner.py`

### Kaggle での実行手順
1. [Kaggle Notebooks](https://www.kaggle.com/code) で新規ノートブックを作成し、`exp2_thinking_partner/exp2_titans_thinking_partner.ipynb` をインポートします。
2. Settings で **Accelerator = GPU T4 x2（または P100）**、**Internet = On** に設定します。
3. デフォルト（`PRESET = 'tiny'`, `SFT_STEPS = 450`, `GRPO_STEPS = 120`, `GROUP_SIZE = 4`, `EVAL_PER_LEVEL = 40`）のまま上から順に実行すると、Stage 1（SFT）→ 評価 → Stage 2（GRPO）→ 比較評価まで一括で回ります。

## 実験2.5: 長距離記憶＆記憶上書きストレステスト

- ノートブック: `exp2_5_memory_stress/exp2_5_titans_memory_stress.ipynb`
- CLI スクリプト: `exp2_5_memory_stress/train_memory_stress.py`

## 実験3 / 3.5: 毎トークン CoT 比較 ＆ フォーマット先行＋長さカリキュラム

従来ターン型（`turn_cot`）・毎トークン残留型（`stream_token_cot`）・毎トークン揮発型（`ephemeral_token_cot`）の 3 方式を Titans (MAC) 上で比較検証した構成です。基礎実験フェーズのまとめは [docs/research_report_20261125.md](docs/research_report_20261125.md) を参照してください。

- Exp 3: `exp3_per_token_cot/exp3_titans_per_token_cot.ipynb`, `exp3_per_token_cot/train_per_token_cot.py`
- Exp 3.5: `exp3_5_length_curriculum/exp3_5_titans_length_curriculum.ipynb`, `exp3_5_length_curriculum/train_per_token_cot_v2.py`
