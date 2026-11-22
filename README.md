# racing_prediction_reboot

## プロダクトの概要
- JRAレースの順位予想をする機械学習モデル
- また、予想順位とオッズ情報を用いて最適馬券購入も計算

## ディレクトリ構成
```
├── configs/                  # 学習・推論のハイパーパラメータ等の設定ファイル
├── data/                     # データ（中身はgit管理外）
│   ├── raw/                  #   取得したままの生データ（不変）
│   ├── external/             #   外部ソースのデータ（オッズ等）
│   ├── interim/              #   前処理途中の中間データ
│   └── processed/            #   学習に使う最終データセット
├── models/                   # 学習済みモデル（git管理外）
├── notebooks/                # 実験・分析用ノートブック
│   ├── predict_jra_racing.ipynb
│   └── optimize_bet.ipynb
├── references/               # データ定義書・参考資料
├── reports/                  # 分析レポート
│   └── figures/              #   レポート用の図
├── src/
│   └── racing_prediction/
│       ├── data/             #   データ取得・生成 (make_dataset.py)
│       ├── features/         #   特徴量エンジニアリング (build_features.py)
│       ├── models/           #   学習・予測 (train_model.py, predict_model.py)
│       ├── betting/          #   最適馬券購入の計算 (optimize_bet.py)
│       └── visualization/    #   可視化 (visualize.py)
└── tests/                    # テストコード
```

## 環境変数の設定
- なしで動くように開発（予定）

## 注意事項
- 本プロダクトは、馬券の当選を保証するものではありません。
- ですので、本プロダクトの利用によりいかなる損害が発生した場合でも、開発者は一切の責任を負いません。
