# GitHub Actions で推論を自動実行する手順

開催日の発走 45 分前ごろに, 各馬の 3 着以内に入る確率とワイドの的中確率をメール / スマホに通知する仕組みの設定手順.

## 全体の流れ

```
[ローカル] データ抽出 (extract_racing_data.ipynb)
     │
     ▼
[ローカル] python scripts/train_on_kaggle.py --update-dataset
     │   データ・コードを Kaggle にアップロード → Kaggle で学習 (predict_jra_racing.ipynb) → 完了を待つ
     │   → 学習済みモデルを models/production/ に取り込む
     ▼
[ローカル] models/production/ を commit して main に push
     ▼
[GitHub Actions] 毎朝 race-day-planner が開催日か判定 → 開催日は race-inference が 10 分おきに予測・通知
```

| ワークフロー | 起動 | 役割 |
|---|---|---|
| `race-day-planner` | 毎日 11:40 (日本時間) | 当日の時刻表を取得し, 対象レースがある日だけ `race-inference` を有効にする. ない日は無効にする |
| `race-inference` | 有効な日の 12:00〜16:50, 10 分おき | 発走 50〜10 分前のレースを予測して通知. 最終レース後に自分を無効にする |

非開催日は `race-inference` が無効になるため起動せず, GitHub Actions の無料枠を節約できる.

---

## 0. 事前準備 (初回のみ)

### Kaggle CLI

```
pip install kaggle
kaggle auth login
```

`kaggle auth login` はブラウザでログインする. ユーザー名 (プロフィール URL の `kaggle.com/<ここ>`) は
スクリプトがログイン情報から自動で読むので, 指定は不要 (表示名を指定すると `Invalid Owner Id` になる).

### データ

`notebooks/extract_racing_data.ipynb` の `collect_races` で, 学習に使う期間のデータを
`data/processed/races/` に抽出しておく.

---

## 1. Kaggle で学習し, モデルを取り込む

リポジトリのルートで実行する.

```
python scripts/train_on_kaggle.py --update-dataset
```

スクリプトが行うこと:

1. `data/processed/races/`, `src/racing_prediction/`, `configs/` を Kaggle Dataset (`<ユーザー名>/racing-prediction-data`, 非公開) にアップロード
   (初回は作成, 2 回目以降は新しいバージョンを追加) し, Kaggle 側の処理が終わるまで待つ
2. 一時フォルダに `predict_jra_racing.ipynb` (出力は消す) と `kernel-metadata.json` を用意
3. `kaggle kernels push` で Kaggle に送り, 最初から最後まで実行 (Web 画面の「Save Version → Save & Run All」と同じ)
4. 1 分ごとに実行状況を確認し, 完了まで待つ (1〜2 時間程度. Kaggle の上限は 12 時間)
5. 出力 (`artifacts/model.txt`, `metadata.json`, `feature_importance.csv`) をダウンロードして `models/production/` に取り込む

最後に学習期間とテストの ROC-AUC / logloss が表示される.

| オプション | 用途 |
|---|---|
| `--update-dataset` | データ・コードをアップロードしてから学習する (通常はこれ) |
| (なし) | データ・コードが前回のアップロードから変わっていないとき |
| `--fetch-only` | 実行済みのノートブックの出力を取り込むだけ (待機中にウィンドウを閉じた場合など) |
| `--dry-run` | 作業用フォルダを作り, 実行するコマンドを表示するだけ |

### 実行結果の保存先

Kaggle での実行結果は, 実行ごとにローカルの `reports/kaggle_runs/<日時>/` に保存される (git 管理外).

```
reports/kaggle_runs/20261123_103000/
├── artifacts/                      学習済みモデル (models/production/ にも取り込まれる)
│   ├── model.txt / metadata.json / feature_importance.csv
├── results/                        評価結果
│   ├── summary.json                設定, 学習・テスト期間, CV / テストの評価指標 (モデルと人気順), 最良パラメータ
│   ├── test_predictions.csv        テスト期間の各馬の予測確率と結果
│   ├── calibration.csv             予測確率帯ごとの実際の 3 着以内率
│   ├── feature_importance_test.csv 特徴量の重要度 (gain / split / SHAP)
│   ├── optuna_trials.csv           Optuna の全試行
│   └── roc_curve.png / feature_importance_top30.png
├── run_log.txt                     実行ログ (print の出力. エラーの行は [stderr] 付き)
└── predict-jra-racing.log          Kaggle の元のログ (JSON)
```

失敗した実行のログは `reports/kaggle_runs/<日時>_error/` に保存され, エラー部分が画面にも表示される.
ローカルでノートブックを実行した場合の評価結果は `reports/latest/` に保存される.

- 実行中にウィンドウを閉じても Kaggle 側の実行は続く. その場合は完了後に `--fetch-only` で取り込む
  (実行結果の保存もこのとき行われる).
- Kaggle での実行が失敗した場合は, スクリプトがログをダウンロードしてエラー部分を表示する.
  詳細は `https://www.kaggle.com/code/<ユーザー名>/predict-jra-racing` の Logs でも確認できる.

### 古い版のデータセットで実行されるのを防ぐ仕組み

Kaggle はデータセットを更新した直後だと, **前の版** のデータセットでノートブックを実行することがある
(古いコード・古い `configs/train.yaml` で学習してしまう). これを防ぐため:

1. アップロードするデータ・コード・設定の中身から識別値 (ハッシュ) を計算し, `dataset_fingerprint.txt` としてデータセットに入れる
2. アップロード後, Kaggle から `dataset_fingerprint.txt` を実際にダウンロードし, 今回の識別値に変わるまで待ってからノートブックを実行する
3. ノートブックには期待する識別値を埋め込んで送り, Kaggle 上で違う版が使われていたら学習せずにエラーで止める

`--update-dataset` なしで実行したときも, Kaggle 上のデータセットがローカルと同じ内容かを確認し,
違えば (設定やコードを変えたのにアップロードしていない場合など) 実行前に止める.

ノートブックが「使われているデータセット (識別値 ...) が, アップロードした最新版 (識別値 ...) ではありません」で
止まった場合は, 数分おいて `python scripts/train_on_kaggle.py` (`--update-dataset` なし) を再実行する.
- 学習データの期間・テストの比率・Optuna の設定は `configs/train.yaml` で指定する (ノートブックの編集は不要).
  設定はデータセットと一緒にアップロードされるので, **設定を変えたら必ず `--update-dataset` を付けて実行する**.

  ```yaml
  data:
    start_date: "2025-11-23"   # 学習・テストに使うデータの開始日 (null なら制限なし)
    end_date: "2026-02-14"     # 終了日 (null なら制限なし)
  split:
    test_ratio: 0.2            # 全開催日数のうち直近の何割をテストにするか
  tuning:
    n_trials: 100              # Optuna の試行回数
    timeout_sec: 3600          # Optuna の打ち切り時間 (秒)
    n_folds: 3                 # 時系列 CV の fold 数
  ```

---

## 2. コードとモデルを main に push する

定期実行 (schedule) は **デフォルトブランチ (main) にあるワークフローだけ** が動く.

```
git add .github src requirements-inference.txt scripts models/production .gitignore docs
git commit -m "Add race inference on GitHub Actions"
git push
```

作業ブランチで作業している場合は, push 後にプルリクエストを作って main にマージする.

---

## 3. 通知先を Secrets に登録する

GitHub のリポジトリ画面 → **Settings → Secrets and variables → Actions → Secrets タブ → New repository secret**

メールとスマホ通知 (ntfy) のどちらか一方は必須 (両方設定すると両方に届く).

### メール (Gmail の例)

| 名前 | 値 |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | 送信に使う Gmail アドレス |
| `SMTP_PASSWORD` | Gmail の **アプリパスワード** (16 文字) |
| `MAIL_TO` | 受信するアドレス (カンマ区切りで複数可. 省略時は `SMTP_USER` に送る) |

アプリパスワード: Google アカウントで 2 段階認証を有効にしてから,「セキュリティ」→「アプリパスワード」で発行する.
通常の Gmail のパスワードでは送信できない.

### スマホのプッシュ通知 (ntfy)

1. スマホに ntfy アプリを入れ, 推測されにくい長いトピック名 (例: `keiba-` + ランダムな英数字 20 文字程度) を購読する
   (トピック名を知っていれば誰でも購読できるため)
2. Secret `NTFY_TOPIC` にそのトピック名を登録する

---

## 4. 対象レースを設定する (任意)

同じ画面の **Variables タブ → New repository variable**

| 名前 | 値の例 | 未設定のとき |
|---|---|---|
| `INFERENCE_VENUES` | `東京` / `東京,中山` | 全競馬場 |
| `INFERENCE_RACE_NUMBERS` | `7-12` / `9,10,11` | `7-12` |

- 競馬場は **名前** で指定する (`05` などのコードは不可).
- 学習データにない競馬場は精度が落ちるため, 学習に使った競馬場に絞るのがおすすめ.

### ワイドの期待値 (`configs/betting.yaml`)

通知には, 推論時点 (発走 45 分前ごろ) のワイドのオッズから計算した **期待値がプラスの組み合わせが全て**
(期待値の高い順に) 入る. 各組み合わせには, モデルの的中確率・オッズ (最低〜最高)・各馬の予測順位を併記する.

```
■ 期待値プラスのワイド (期待値の高い順)
  4組 / 購入額 400円 (各100円の場合)
⑤-⑪ 期待値+47%
  モデル確率 6.2% / オッズ 47.4〜52.5倍
  スナッピードレッサ(予測1位) / アスクアイルビゼア(予測8位)
...
```

期待値がプラスの組み合わせがなければ「見送り推奨」と表示する. 件名にも期待値がプラスの組数が入る.

- 期待値 = 確率 × **最低オッズ** − 1 (ワイドのオッズは幅があるため, 控えめな最低オッズで計算)
- 確率は, モデルの確率と **オッズから逆算した市場の確率** を組み合わせたもの (そのため, モデル確率 × オッズとは一致しない).
  モデルの確率だけで計算すると, モデルが過大評価している大穴 (オッズ数百倍) の期待値が異常に大きくなり,
  実際にはマイナスの馬券を勧めてしまうため.
- オッズが高すぎる組み合わせ・モデルの確率が低すぎる組み合わせは, モデルを信用せず市場の確率だけで計算する
  (期待値は控除率の分マイナスになるので, 一覧には出ない).

| 設定 | 既定値 | 意味 |
|---|---|---|
| `wide.market_weight` | 0.5 | 市場の確率をどれだけ重視するか (0 = モデルだけ, 1 = 市場だけ) |
| `wide.max_odds` | 50 | 最低オッズがこれより高い組み合わせは市場の確率だけで計算 |
| `wide.min_model_prob` | 0.05 | モデルのワイド的中確率がこれより低い組み合わせも市場の確率だけで計算 |

設定を変えたら commit して main に push する (GitHub Actions はリポジトリの `configs/betting.yaml` を読む).

**注意**: 現在のモデルはテストで人気順 (オッズ) より精度が低いため, 「期待値プラス」でも実際に利益が出るとは限らない.
`market_weight` は 0.5 以上のまま運用し, 過去レースでの検証 (ワイドの払戻金を使った回収率の確認) を行ってから
実際の購入に使うことを推奨する.

---

## 5. Actions の権限を確認する

**Settings → Actions → General**

- **Actions permissions**: 「Allow all actions and reusable workflows」になっていること
- **Workflow permissions**: 変更不要 (各ワークフローがファイル内で `actions: write` を指定している)

---

## 6. 動作確認

1. **race-day-planner を手動実行**
   Actions タブ → `race-day-planner` → **Run workflow**.
   ログの「Enable or disable race-inference」に有効化 / 無効化の結果が出る.
   ここでエラーになる場合は, ワークフローを切り替える権限が足りていない (手順 5 を確認).

2. **race-inference を通知なしで試す**
   Actions タブ → `race-inference`. 無効になっていれば先に **Enable workflow** を押す.
   **Run workflow** で `race_id` に過去のレース (例: `202605050511`) を入れ, `dry_run` にチェックして実行.
   ログに予測結果の本文が出れば OK.
   netkeiba からの取得で失敗する場合は, GitHub のサーバーからのアクセスがブロックされている可能性がある.

3. **通知が届くか試す**
   `dry_run` のチェックを外してもう一度実行し, メール / スマホに届くことを確認する.

以上で設定完了. 以降は開催日に自動で通知が届く.

---

## 運用

| やりたいこと | 方法 |
|---|---|
| モデルを作り直す | データを抽出 → `python scripts/train_on_kaggle.py --update-dataset` → `models/production` を commit して main に push |
| 特定のレースを手動で予測 | `race-inference` を Enable → Run workflow で `race_id` を指定 |
| 一時的に止める | Actions 画面で `race-day-planner` と `race-inference` を **Disable workflow** |
| 通知が来ないとき | Actions タブで `race-day-planner` / `race-inference` のログを確認 (失敗した実行は GitHub からメールでも通知される) |

## 注意点

- GitHub Actions の定期実行は混雑時に数分〜十数分遅れることがある. 予測の範囲 (発走 50〜10 分前) を広めにとって吸収している.
- `race-inference` が無効の間は手動実行 (Run workflow) もできない. 先に Enable workflow を押す.
- 公開リポジトリは 60 日間コミットがないと定期実行が自動で止まる. 止まったら Actions 画面で有効に戻す.
- 非公開リポジトリの無料枠は月 2,000 分. 目安は月 250 分程度 (開催日 8〜9 日の場合).
- 予測は参考情報であり, 馬券の的中を保証するものではない.
