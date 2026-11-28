## 競馬予想モデルの要件定義

### 環境
- カーネル：kaggleのカーネルを仕様（スペックの高いリソースを使いたい）
- データフレーム：pandas（データ量が数万〜数十万行程度のため。Kaggle は単一マシンで Spark の分散の恩恵がない）

### モデル
- light-gbm
- optunaでハイパラチューニングもしたい

### 条件
- 各レースごとに、各馬が1着になる確率を予測するモデルを作りたい。（主用途はワイドの予想）
  - レースごとの出走馬が1着になる確率の総和は必ず1になるようにする。
    → 条件付きロジット（レース内ソフトマックス）: LightGBM が各馬の強さの対数を出力し、レース内でソフトマックスをとって1着の確率にする。
      学習では1着の馬に付けた確率の -log（レース単位の logloss）を最小化する（LightGBM のカスタム目的関数）。
  - ワイド（2頭とも3着以内）の確率は、1着の確率を各馬の強さとする Plackett-Luce モデル（Harville モデル）で着順をシミュレーションして求める。
  - 精度評価では ROC-AUC（1着か否か）を出し、確定単勝オッズから逆算した確率で予想した場合と比較する。
  - あわせて「スコアがトップの馬が1着になった割合」と「単勝オッズが一番低い馬（1番人気）が1着になった割合」を比較する（差の95%信頼区間・マクネマー検定。各レースでそれぞれの馬の単勝を100円ずつ買った場合の総利益と回収率も比較する。predict_model.compare_top_pick。ノートブックと models/production/METRICS.md に表示）。
  - オッズに関する情報（今回の単勝オッズ・人気、過去のレースの人気・単勝オッズ）は特徴量に使わない。
  - オッズとの差だけを学習する（configs/train.yaml の model.market_offset。既定は true）: 単勝オッズから逆算した確率の対数を各馬のスコアの出発点（LightGBM の init_score）にし、木は「オッズがどれだけ外れているか」だけを学習する。予測時も同じ値を足す（predict_model.predict_win）。学習には確定オッズ、推論では発走 12 分前のオッズを使う。false にするとオッズを使わないモデル（以前の方式）になる。
  - スコア（1着の確率）の低い順に馬を並べて頭数が同じになるように10等分し、帯ごとにスコアの平均と実際の1着率（95%信頼区間つき）を比べる（predict_model.compare_calibration。method="width" で10%ごとの等間隔の帯にもできる。単勝オッズから逆算した確率と並べる。ECE = 帯ごとの |差| を頭数で重み付けした平均）。
  - **メインの目標: モデルを混ぜるとオッズ単体より当たること**（predict_model.compare_with_market）。a × log(モデルの確率) + b × log(オッズの確率) をレース内ソフトマックスした予測（a, b は最尤推定。テスト期間の前半・後半で交差して評価）か、モデル単体の logloss が、オッズ単体（そのままと補正後の良い方）より小さく、改善の 95% 信頼区間が 0 より上なら達成。ノートブックの 4 章と models/production/METRICS.md の先頭に表示する。
- 当然各馬の強さだけでなく、ほかの馬の強さ、当日の場体重・馬場状態などにも左右されるので、それらを考慮して予測できるモデルを作りたい。
- 馬体重はレース60分前に発表されるので、その発表後（余裕見てレース開始45分前に）推論を実行し、結果をメールなどで受け取りたい。
  - 出先なので可能なら、PCを持ち歩かずにスマホで結果を受け取れることがうれしいが、ネックはおそらくkaggleのカーネルが長時間動かしていないとスリープになること

### データ
- data/processed/races以下にあるcsvから特徴量・目的変数を取得
- データはローカル、学習の実行がkaggleになるので、それでも実現できるようにしたい。
- 平均順位について、特にレース数が少ない馬だとNaNが多発するのでモデルを作るときはそこに注意したい。

### 特徴量・目的変数など
特徴量
- レースの場所 (race_venue)
- 距離区分 (race_distance_category)：short=1000〜1400m / mid=1500〜2400m / long=2500m以上
- 芝・ダート (race_surface)：turf / dirt
- 当日の馬場状態 (race_track_condition)：good=良 / yielding=稍重 / soft=重 / heavy=不良
- 出走頭数 (n_horses)：取消・除外を除く
- 距離 (race_distance)：m
- クラス (race_class)：0=新馬・未勝利 / 1〜3=1〜3勝クラス / 4=オープン・L / 5〜7=G3〜G1（出馬表のグレードのアイコンとクラス欄から判定）
- 1着の本賞金 (race_prize)：万円（クラスの高さを連続値で表す）
- 枠番 (waku)
- 性別 (sex)
- 馬齢 (age)
- 斤量 (weight_carried)
- 馬体重 (horse_weight)
- 馬体重の増減 (weight_diff)
- 芝全体の平均順位 (avg_rank_turf)
- 芝の距離別の平均順位 (avg_rank_turf_short / _mid / _long)
- 芝の馬場状態別の平均順位 (avg_rank_turf_good / _yielding / _soft / _heavy)
- 芝の競馬場別の平均順位 (avg_rank_turf_sapporo / _hakodate / _fukushima / _niigata / _tokyo / _nakayama / _chukyo / _kyoto / _hanshin / _kokura)
- 芝の海外レースでの平均順位 (avg_rank_turf_overseas)
- ダート全体の平均順位 (avg_rank_dirt)
- ダートの距離別の平均順位 (avg_rank_dirt_short / _mid / _long)
- ダートの馬場状態別の平均順位 (avg_rank_dirt_good / _yielding / _soft / _heavy)
- ダートの競馬場別の平均順位 (avg_rank_dirt_sapporo / _hakodate / _fukushima / _niigata / _tokyo / _nakayama / _chukyo / _kyoto / _hanshin / _kokura)
- ダートの海外レースでの平均順位 (avg_rank_dirt_overseas)
- 条件ごとの出走回数 (n_starts_*)：上記の平均順位 avg_rank_* と同じ条件ごとに、平均の計算に使ったレース数（該当なしは 0）
- 近走・能力 (hist_*)：戦績ページの 頭数・騎手・斤量・タイム・着差・通過順位・上がり・賞金・レース名(クラス) から計算
  （対象レースの開催日より前のレースだけを使う。一覧と意味は src/racing_prediction/features/build_features.py の HISTORY_FEATURE_NAMES）
  - 通算・直近1年の出走数、前走からの日数、通算の勝率・3着以内率、通算・直近1年の獲得賞金
  - 近走の成績：前走・直近3走・5走の 着順/頭数、着差（秒）
  - 上がり3F（前走・直近3走の最速）、脚質（直近3走の最初・最後のコーナーの位置 / 頭数）、直線での追い上げ
  - 前走からの変化：クラス（昇級）、距離、芝・ダート、騎手の乗り替わり、斤量
- スピード指数 (speed_last1 / speed_best3 / speed_avg3)：直近3走の走破タイムが、コース・距離・馬場状態ごとの基準タイムより
  1000m あたり何秒速いか。基準タイムは学習データの各馬の直近3走の中央値（前処理の fit で決める）
- 騎手・調教師・父・母父の成績 ({jockey_name, trainer_name, sire, dam_sire}_win_rate / _top3_rate / _n_runs)：
  対象レースの開催日より前の学習データのレースでの 勝率・3着以内率・出走数（全体の平均を20走分足して割り、少ない出走数のぶれを抑える）。
  名前をそのままカテゴリ変数にすると、種類が多すぎて丸暗記（過学習）するため置き換えた
- レース内の逃げ・先行馬の頭数 (race_n_front_runners)：展開の指標
- レース内での比較 (*_race_rank / *_race_diff)：平均着順・近走の成績・スピード指数・騎手の勝率などの、同じレースの出走馬の中での順位と平均との差
- 父の系統 (sire_line)

目的変数
- 1着か否か (is_win)：1着なら1、それ以外は0（CSV には保存せず、着順 rank から作る。同着の場合は1着の馬が複数）

評価用の列
- 3着以内か否か (is_top3)：3着以内なら1、それ以外は0（ワイドの評価に使う。CSV に保存する）

特徴量に使わない列
- レースID・開催日・馬番・馬名 (race_id, race_date, umaban, horse_name)：識別用（馬名は丸暗記による過学習を防ぐため特徴量にしない）
- 着順 (rank)：目的変数 (is_win) の元になる値
- 単勝オッズ・人気 (win_odds, popularity)：使わない（確定オッズは精度の比較に使う）
- 過去のレースの人気・単勝オッズ (hist_last1_popularity, hist_avg3_popularity, hist_avg5_rank_minus_pop, hist_last1_log_odds, hist_avg3_log_odds)：
  CSV には保存するが、オッズに関する情報なので使わない（model_features.ODDS_COLUMNS）
- 騎手名・調教師名・父・母父 (jockey_name, trainer_name, sire, dam_sire)：そのままは使わず、上の成績（勝率など）の計算に使う
- 母・父父・父母・母母 (dam, sire_sire, sire_dam, dam_dam)：1頭ごとにほぼ違い成績を出せないため使わない（CSV には保存する）
- 馬のID (horse_id)：戦績の取得用
- 直近3走のタイム・距離・芝ダ・競馬場・馬場状態 (hist_last{1〜3}_*)：スピード指数の計算用