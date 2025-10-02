# NTT株価予測モデル
## 概要
本リポジトリでは、NTTの株価データを用いて時系列予測モデルを構築しています。  
主に以下のモデルを実装しています：
- ARIMAモデルによる株価予測
- Prophetモデルによる株価予測（出来高を特徴量として使用）

データの前処理、特徴量エンジニアリング、モデル学習、予測精度評価までを含みます。

## データ
- 使用データ: `stock_price.csv`（NTT株価データ）
- データ内容: 日付、終値、始値、高値、安値、出来高など

※ CSV ファイルはリポジトリに含めるか、別途提供してください。

## 必要なライブラリ
- Python 3.x
- 主なライブラリ:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - statsmodels
  - prophet
  - yfinance

依存関係は `requirements.txt` にまとめています。

##実行方法
1. リポジトリをクローン：
```bash
git clone https://github.com/nao70-mori/ntt-stock-prediction.git
cd ntt-stock-prediction
```
2. 必要なライブラリをインストール：
```bash
pip install -r requirements.txt
```
3.プログラムを実行：
```bash
python NTT_stock_prediction.py
```

