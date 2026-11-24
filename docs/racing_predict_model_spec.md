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
- 枠番 (waku)
- 性別 (sex)
- 馬齢 (age)
- 斤量 (weight_carried)
- 騎手名 (jockey_name)
- 調教師名 (trainer_name)
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
- 父 (sire)
- 父の系統 (sire_line)
- 母 (dam)
- 父父 (sire_sire)
- 父母 (sire_dam)
- 母父 (dam_sire)
- 母母 (dam_dam)

目的変数
- 1着か否か (is_win)：1着なら1、それ以外は0（CSV には保存せず、着順 rank から作る。同着の場合は1着の馬が複数）

評価用の列
- 3着以内か否か (is_top3)：3着以内なら1、それ以外は0（ワイドの評価に使う。CSV に保存する）

特徴量に使わない列
- レースID・開催日・馬番・馬名 (race_id, race_date, umaban, horse_name)：識別用（馬名は丸暗記による過学習を防ぐため特徴量にしない）
- 着順 (rank)：目的変数 (is_win) の元になる値
- 確定単勝オッズ (win_odds)：馬券購入の最適化・精度評価での比較用。特徴量に使うとリークになる
- 人気 (popularity)：馬券購入の最適化用。特徴量に使うとリークになる