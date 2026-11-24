# GitHub Actions で推論を自動実行する手順

開催日の **発走 12 分前** に, その時点のワイドのオッズで計算した期待値プラスのワイド ボックスと,
各馬の 1 着になる確率 (と, そこから計算した 3 着以内に入る確率) をメール / スマホに通知する仕組みの設定手順.

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
[GitHub Actions] 毎朝 race-day-planner が開催日か判定 → 開催日は race-inference を起動 → 常駐して各レースの発走 12 分前に推論・通知
```

| ワークフロー | 起動 | 役割 |
|---|---|---|
| `race-day-planner` | 毎日 8:17, 8:47 (日本時間) | 当日の時刻表を取得し, 対象レースがある日だけ `race-inference` を有効にして起動する. ない日は無効にする |
| `race-inference` | `race-day-planner` が起動 (予備として有効な日の 9:07〜16:37 に 30 分おき) | 常駐し, 各レースの発走 12 分前に推論を始め (その時点の単勝オッズを特徴量に使う. 1〜2 分), 続けてその時点のオッズで期待値を計算して通知 (1 レース 1 通). 単勝オッズが取れなければ再試行する. 5 時間 30 分で次の実行に引き継ぐ. 最終レース後に自分を無効にする |

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

- 競馬場は複数指定できる (`venues = ["東京", "中山", "京都", "阪神"]`). CSV は競馬場ごとのフォルダ
  (`data/processed/races/tokyo/`, `kyoto/` など) に保存される.
- 学習は `data/processed/races/` 以下の全競馬場の CSV を使う. 使う競馬場を絞るときは `configs/train.yaml` の
  `data.venues` に書く (例: `venues: [東京, 中山]`. `null` なら全競馬場).
- 通知する競馬場は, これとは別に GitHub の Variables の `INFERENCE_VENUES` で決める (下の「設定できる値」).
  学習に使っていない競馬場のレースも予測はできるが, 学習データにある競馬場のほうが精度が出やすい.

- 一度取得したページは `data/raw/` に保存し, 次からはアクセスせずに使う (git 管理外. どのフォルダから実行しても同じ場所).
  - 保存済みのレース (`data/processed/races/` の CSV): 取得しない (`overwrite=True` のときだけ取り直す)
  - 各馬の血統 (`data/raw/horse_pedigree/`): 変わらないので取り直さない
  - 開催カレンダー (`data/raw/race_calendar/`, 月ごと) とレース一覧 (`data/raw/race_list/`, 開催日ごと):
    終わった月・開催日の分は取り直さない (今月・今日以降は開催の変更があり得るので毎回取得する)
  - 各馬の戦績 (`data/raw/horse_results/`): 保存した日より前のレースに使うときは取り直さない
    (保存した後のレースに使うときは, その後に走ったレースを反映するため取り直す)
- 列 (特徴量) を追加する前に保存した CSV は, 学習で読み飛ばされる. 次のコマンドで足りない列を補う
  (血統などの保存済みの列はそのまま使い, 出馬表と戦績だけを取り直す. 800 レースで 3〜5 時間. 途中で止めても続きから再開できる).
  `collect_races` で同じ期間を取り直した場合も, 古い CSV には同じように列が補われる.

  ```
  python scripts/backfill_race_features.py               # 列が足りない CSV だけを補う
  python scripts/backfill_race_features.py --recompute   # 特徴量の計算方法を変えたとき (保存済みの戦績から計算し直す)
  ```

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

Kaggle での実行結果は, ローカルの `reports/kaggle_runs/` に **学習した日付ごとのフォルダ** を作り,
その下に **学習の実行ごとに, モデルのバージョン名のフォルダ** で保存される (git 管理外).
バージョン名は `v{学習した日付_時刻}` (日本時間. 例: `v20261124_153012`).

```
reports/kaggle_runs/
├── 20261124/                           学習した日付
│   ├── v20261124_153012/               1 回目の学習 (バージョン名)
│   ├── v20261124_201845/               2 回目の学習
│   └── error_210512/                   失敗した実行のログ (error_<時刻>)
└── 20261125/
    └── v20261125_093001/
```

各バージョンのフォルダの中身:

```
v20261124_153012/
├── artifacts/                      学習済みモデル (models/registry/ に登録される)
│   ├── model.txt / metadata.json / feature_importance.csv
├── results/                        評価結果
│   ├── summary.json                設定, 学習・テスト期間, CV / テストの評価指標 (モデルと単勝オッズ), 最良パラメータ
│   ├── test_predictions.csv        テスト期間の各馬の予測確率と結果
│   ├── calibration.csv             予測確率帯ごとの実際の 1 着率
│   ├── feature_importance_test.csv 特徴量の重要度 (gain / split / SHAP)
│   ├── optuna_trials.csv           Optuna の全試行
│   └── roc_curve.png / feature_importance_top30.png
├── run_log.txt                     実行ログ (print の出力. エラーの行は [stderr] 付き)
└── predict-jra-racing.log          Kaggle の元のログ (JSON)
```

失敗した実行のログは `reports/kaggle_runs/<日付>/error_<時刻>/` に保存され, エラー部分が画面にも表示される.
同じ実行の出力を取り込み直した場合 (`--fetch-only` など) は, 同じバージョンのフォルダを置き換える (二重に登録しない).
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
git add .github src requirements-inference.txt scripts models .gitignore .gitattributes configs docs
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
| `NOTIFY_MINUTES_BEFORE` | `12` / `12,7` | `12` (発走何分前のオッズで通知するか. カンマ区切りで複数も可) |

- 競馬場は **名前** で指定する (`05` などのコードは不可).
- 学習データにない競馬場は精度が落ちるため, 学習に使った競馬場に絞るのがおすすめ.

### ワイド ボックスの期待値 (`configs/betting.yaml`)

通知には, 発走 12 分前 (`NOTIFY_MINUTES_BEFORE`) の時点のワイドのオッズから, **出走馬で作れる全てのボックス (2 頭〜全頭) の期待値**
を計算し, **期待値がプラスのボックス** を **的中率の高い順** に入れる (18 頭立てなら約 26 万通りを計算する).

```
■ 期待値プラスのワイド ボックス (的中率の高い順)
  全65,519通り中 120通りが期待値プラス (上位20通りを表示)
 1. ⑤⑦⑪ 3頭 3点/300円
    期待払戻311円 回収率104% 的中率25%
...
```

- ボックスの期待払戻 = ボックスに含まれる各組の (確率 × **最低オッズ** × 100 円) の合計
  (ワイドのオッズは幅があるため, 控えめな最低オッズで計算. 期待値は足し算できるので, 組同士が独立でなくても正確)
- 的中率 = ボックス内の 2 頭以上が 3 着以内に入る確率 (モデルの 1 着の確率を各馬の強さとして着順をシミュレーションして計算).
  期待値がプラスの全ボックスについて計算し, 的中率の高い順 (同じなら期待回収率の高い順) に並べる.
  頭数の多いボックスほど的中率は高くなるが, 点数 (購入額) も増える点に注意.
- 期待値がプラスのボックスがなければ「見送り推奨」と表示する. 件名にも期待値がプラスのボックスの数が入る.
- 確率は, モデルの確率と **オッズから逆算した市場の確率** を組み合わせたもの (`wide.market_weight`).
  モデルの確率だけで計算する (`market_weight: 0`) と, モデルが過大評価している大穴 (オッズ数百倍) を含むボックスの
  期待値が異常に大きくなるため注意.

| 設定 | 既定値 | 意味 |
|---|---|---|
| `wide.market_weight` | 0.5 | 市場の確率をどれだけ重視するか (0 = モデルだけ, 1 = 市場だけ) |
| `wide.max_odds` | 50 | 最低オッズがこれより高い組み合わせは市場の確率だけで計算 |
| `wide.min_model_prob` | 0.05 | モデルのワイド的中確率がこれより低い組み合わせも市場の確率だけで計算 |
| `box.max_display` | 20 | メールに載せる期待値プラスのボックスの最大数 (的中率の高い順. 総数は常に表示) |

設定を変えたら commit して main に push する (GitHub Actions はリポジトリの `configs/betting.yaml` を読む).

**注意**: モデルは単勝オッズも特徴量に使う (学習は確定オッズ, 推論は発走 12 分前のオッズ). テストで単勝オッズだけの予想を
上回っているかは `models/production/METRICS.md` で確認できる. 上回っていなければ, 「期待値プラス」でも実際に利益が出るとは限らない.
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

## モデルのバージョン管理

推論で使うモデルはバージョン管理しており, 精度が下がったときに以前のモデルに戻せる (`scripts/model_registry.py`).

```
models/registry/
├── registry.json            各バージョンの学習期間・テストの精度 (モデルと単勝オッズ) と, 適用の履歴
├── v20261124_153012/        各バージョン (v{学習した日付_時刻})
│   ├── model.txt / metadata.json / feature_importance.csv   モデル
│   └── summary.json / roc_curve.png / calibration.csv       テストの評価結果 (ROC-AUC, ROC 曲線など)
└── v20261125_093001/
models/production/           今適用しているバージョンのコピー (GitHub Actions の推論はここを読む)
├── model.txt / metadata.json / feature_importance.csv
├── summary.json / roc_curve.png / calibration.csv   適用中のモデルのテストの評価結果
├── METRICS.md               適用中のモデルの精度のまとめ (ROC-AUC などの単勝オッズとの比較表と ROC 曲線. GitHub の画面で読める)
└── VERSION.txt              今適用しているバージョン名
```

- バージョン名は `v{学習した日付_時刻}` (日本時間). 学習時に `metadata.json` の `trained_at` に記録した日時から決める.

- **既定 (自動モード) では, 登録済みの全バージョンのうちテストの ROC-AUC が最良のモデルが GitHub Actions の推論に使われる.**
  `train_on_kaggle.py` で学習すると, 新しいモデルは新しいバージョンとして自動で登録され, そのたびに
  ROC-AUC が最良のバージョンを選び直して `models/production/` に適用する (新しいモデルの方が低ければ, 今のモデルのまま).
- `promote` / `rollback` で手動で選んだ場合は **手動モード** になり, そのモデルに固定される
  (新しいモデルを取り込んでも切り替わらない). `best` で自動モードに戻る.
- 通知メールの末尾に, 使ったモデルのバージョン・学習期間・テストの ROC-AUC が表示される.
  例: `※ モデル: v20261124_153012 (学習期間 2025-11-23〜2026-11-22 / テストのROC-AUC 0.758 (レース内 0.752))`
- `train_on_kaggle.py` のオプション: `--force-promote` (ROC-AUC に関係なく新しいモデルを適用. 手動モード) / `--no-promote` (登録だけ)

| コマンド | 内容 |
|---|---|
| `python scripts/model_registry.py list` | バージョンの一覧 (`*` が適用中, `★` が ROC-AUC 最良), 適用モード, 適用の履歴 |
| `python scripts/model_registry.py best` | 自動モードに戻し, ROC-AUC が最良のバージョンを適用する |
| `python scripts/model_registry.py rollback` | 1 つ前に適用していたモデルに戻す (続けて実行すると, さらに前へ戻る. 手動モード) |
| `python scripts/model_registry.py promote v20261124_153012` | 指定したバージョンを適用する (手動モード) |
| `python scripts/model_registry.py register <フォルダ>` | 手動でダウンロードしたモデルを登録する |

適用 (`promote` / `rollback`) したら, GitHub Actions に反映するため `models` を commit して main に push する.

**注意**: 自動の選択は, 各モデルの学習時のテスト期間の ROC-AUC どうしを比べている. 学習期間が違うとテスト期間も違うため,
厳密な比較ではない (テスト期間の短いモデルは数値がぶれやすい). 一覧の「テスト期間」も見て判断すること.

---

## 運用

| やりたいこと | 方法 |
|---|---|
| モデルを作り直す | データを抽出 → `python scripts/train_on_kaggle.py --update-dataset` → `models` を commit して main に push |
| 精度が下がったので前のモデルに戻す | `python scripts/model_registry.py rollback` → `models` を commit して main に push |
| ROC-AUC が最良のモデルを使う (自動モードに戻す) | `python scripts/model_registry.py best` → `models` を commit して main に push |
| モデルの一覧・精度を見る | `python scripts/model_registry.py list` |
| 特定のレースを手動で予測 | `race-inference` を Enable → Run workflow で `race_id` を指定 |
| 一時的に止める | Actions 画面で `race-day-planner` と `race-inference` を **Disable workflow** |
| 通知が来ないとき | Actions タブで `race-day-planner` / `race-inference` のログを確認 (失敗した実行は GitHub からメールでも通知される) |

## 注意点

- GitHub Actions の定期実行は混雑時に数十分起動しないことがある. そのため `race-inference` は定期実行の時刻に頼らず,
  1 回の実行の中で各レースの発走 12 分前まで待って推論・通知する (ジョブの上限 6 時間の手前で, 次の実行を自分で起動して引き継ぐ).
- `race-inference` が無効の間は手動実行 (Run workflow) もできない. 先に Enable workflow を押す.
- 公開リポジトリは 60 日間コミットがないと定期実行が自動で止まる. 止まったら Actions 画面で有効に戻す.
- 待機している時間も実行時間に数えられる (開催日は 1 日 7〜8 時間). 公開リポジトリは無料.
  非公開リポジトリでは無料枠 (月 2,000 分) を超えるため, 公開リポジトリで運用する.
- 推論は発走 12 分前に始めるので, メールが届くのは 1 レースあたり推論にかかる 1〜2 分後 (発走 10 分前頃).
  同じ時刻帯に複数のレースがある場合は 1 レースずつ順に処理するため, 後のレースは数分遅れることがある.
  起動の遅れなどでタイミングを過ぎていても, 発走 1 分前までなら推論して送る (発走後は送らない).
  推論・通知に失敗したときは 3 回まで試す.
- 予測は参考情報であり, 馬券の的中を保証するものではない.
