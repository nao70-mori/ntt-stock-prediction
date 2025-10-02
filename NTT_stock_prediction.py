import pandas as pd
import matplotlib.pyplot as plt
# Mac標準の日本語フォントを指定
plt.rcParams['font.family'] = 'Hiragino Sans'
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# CSV読み込み
df = pd.read_csv("stock_price.csv", encoding="utf-8")

# データの先頭5行確認
df.head()
print(df.head())

# データ型確認
df.info()
print(df.info())

# 日付を datetime 型に変換
df['日付け'] = pd.to_datetime(df['日付け'])

# 出来高の末尾1文字を集計（数字だけのものは例外処理）
df["単位"] = df["出来高"].str.extract(r"([A-Za-z]+)$")  # 末尾のアルファベットを抽出
print(df["単位"].value_counts())

def convert_volume_to_M(x):
    try:
        if x.endswith("M"):
            return float(x[:-1])
        elif x.endswith("B"):
            return float(x[:-1]) * 1000
        else:
            return float(x)/1e6  # 単位なしの場合を百万単位に変換
    except:
        return np.nan


df["出来高_M"] = df["出来高"].apply(convert_volume_to_M)

df['変化率 %'] = df['変化率 %'].str.replace('%','').astype(float)

# 終値、始値、高値、安値、出来高、変化率の統計量
df.describe()
print(df.describe())

plt.figure(figsize=(12,6))
plt.plot(df['日付け'], df['終値'], label='終値')
plt.title('NTT株価（終値）推移')
plt.xlabel('日付')
plt.ylabel('株価（円）')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(df['日付け'], df['変化率 %'], label='変化率', color='orange')
plt.title('NTT株価 日次変化率')
plt.xlabel('日付')
plt.ylabel('変化率 (%)')
plt.grid(True)
plt.show()

# 移動平均を計算
df['MA5'] = df['終値'].rolling(window=5).mean()
df['MA25'] = df['終値'].rolling(window=25).mean()

# プロット
plt.figure(figsize=(12,6))
plt.plot(df['日付け'], df['終値'], label='終値', alpha=0.5)
plt.plot(df['日付け'], df['MA5'], label='5日移動平均')
plt.plot(df['日付け'], df['MA25'], label='25日移動平均')

plt.title('NTT株価（終値と移動平均）')
plt.xlabel('日付')
plt.ylabel('株価（円）')
plt.legend()
plt.grid(True)
plt.show()

# 月ごとに抽出
df['月'] = df['日付け'].dt.month

# 月ごとの平均変化率を計算
monthly_mean = df.groupby('月')['変化率 %'].mean()

print("月ごとの平均変化率（%）")
print(monthly_mean)

# 可視化（棒グラフ）
plt.figure(figsize=(10,5))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title("月ごとの平均変化率")
plt.xlabel("月")
plt.ylabel("平均変化率 (%)")
plt.grid(axis='y')
plt.show()

# ヒストグラム
plt.figure(figsize=(10,5))
plt.hist(df["変化率 %"], bins=50, edgecolor="black")
plt.title("変化率の分布（ヒストグラム）")
plt.xlabel("変化率 (%)")
plt.ylabel("頻度")
plt.show()

# 箱ひげ図
plt.figure(figsize=(6,5))
plt.boxplot(df["変化率 %"].dropna(), vert=True)
plt.title("変化率の分布（箱ひげ図）")
plt.ylabel("変化率 (%)")
plt.show()

threshold = 5  #±5%以上を異常とする(±10%が限界値であり、±5%は大きめの変動のため異常値にした)
outliers = df[abs(df["変化率 %"]) > threshold][["日付け","変化率 %"]]
print(outliers)


df["変化率 % clipped"] = df["変化率 %"].clip(-threshold, threshold)
print(df["変化率 % clipped"])

df['終値_clipped'] = df['終値'].copy()
df['変化率_clipped'] = df['変化率 %'].clip(-threshold, threshold)

# 前日終値と clipped 変化率から新しい終値を計算
df['終値_clipped'] = df['終値'].shift(1) * (1 + df['変化率_clipped']/100)



df["年"] = df["日付け"].dt.year
monthly_trend = df.groupby(["年","月"])["変化率 %"].mean().unstack()
monthly_trend = monthly_trend.fillna(0)
print(monthly_trend)

# 全体の月ごとの平均
overall_monthly_trend = monthly_trend.mean()
print(overall_monthly_trend.to_string())

# 可視化
overall_monthly_trend.plot(kind="bar", figsize=(10,5), color="skyblue")
plt.title("全期間の月ごとの平均変化率")
plt.xlabel("月")
plt.ylabel("平均変化率 (%)")
plt.grid(axis="y")
plt.show()



df_arima = df[['日付け', '終値_clipped']].copy()
df_arima['日付け'] = pd.to_datetime(df_arima['日付け'])
df_arima = df_arima.set_index('日付け')  # ← これが必要
df_arima = df_arima.asfreq('B')          # 営業日ベースに
df_arima['終値_clipped'] = df_arima['終値_clipped'].ffill()



# スケーリング対象の列を指定（MA5, MA25 を除外）
features_to_scale = ["終値", "始値", "高値", "安値", "出来高_M", "変化率 %"]

scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print(df[features_to_scale].head())

# 確認
print(df_arima.head())

# 学習データとテストデータに分割（80%学習, 20%テスト）
train_size = int(len(df_arima) * 0.8)
train, test = df_arima.iloc[:train_size], df_arima.iloc[train_size:]

# ARIMA モデル構築（p,d,q は例として設定）
model = ARIMA(train['終値_clipped'], order=(1,2,1)) 
# 簡易設定 
model_fit = model.fit()
print(model_fit.summary())



# まずは行番号ベースで予測（従来通り）
# 日付インデックスを使って予測
test_pred = model_fit.predict(
    start=len(train), 
    end=len(train)+len(test)-1, 
    typ='levels'
)

# そのあと日付インデックスを test.index に揃える
test_pred = pd.Series(test_pred, index=test.index)

train_pred_vals = model_fit.predict(start=0, end=len(train)-1, typ='levels')
train_pred = pd.Series(train_pred_vals, index=train.index)

mask = ~test['終値_clipped'].isna() & ~test_pred.isna()
y_true = test['終値_clipped'][mask]
y_pred = test_pred[mask]

if len(y_true) == 0:
    print("RMSE 計算対象のデータがありません。")
else:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    print(f"テストデータ RMSE: {rmse:.4f}")

mae = mean_absolute_error(y_true, y_pred)

print(f"テストデータMAE: {mae:.4f}")


print(test_pred.head(10))
print(test_pred.isna().sum())


# プロットで比較
plt.figure(figsize=(12,6))
plt.plot(train.index, train['終値_clipped'], label='Train Actual', color='red')
plt.plot(test.index, test['終値_clipped'], label='Test Actual', color='orange')
plt.plot(test.index, test_pred, label='Test Predicted', color='green')
plt.title('ARIMAによるNTT株価予測（終値）')
plt.xlabel('日付')
plt.ylabel('終値')
plt.legend()
plt.grid(True)
plt.show()

print(train.head())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 出来高の文字列を数値に変換する関数 ===
def parse_volume(x):
    """'1.28B' などの文字列を数値に変換"""
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        if x.endswith("K"):
            return float(x[:-1]) * 1e3
        elif x.endswith("M"):
            return float(x[:-1]) * 1e6
        elif x.endswith("B"):
            return float(x[:-1]) * 1e9
        else:
            return float(x)
    return x

# === 出来高を数値に変換 ===
df['出来高'] = df['出来高'].apply(parse_volume)

# === Prophet用データ整形 ===
df_prophet = df[['日付け', '終値_clipped', '出来高']].rename(
    columns={'日付け': 'ds', '終値_clipped': 'y'}
)

# NaNを削除
df_prophet = df_prophet.dropna(subset=['y'])

# 日付を昇順にソート
df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

# === 学習・テスト分割（80%学習, 20%テスト） ===
train_size = int(len(df_prophet) * 0.8)
train_p, test_p = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]

# === Prophet モデル構築 ===
prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True)

# 出来高を追加
prophet_model.add_regressor('出来高')

# 学習
prophet_model.fit(train_p)

# === 未来データフレーム作成（テスト期間分） ===
future = prophet_model.make_future_dataframe(periods=len(test_p), freq='B')

# 出来高をマージ
future = future.merge(df_prophet[['ds', '出来高']], on='ds', how='left')

# NaN を埋める（直前値で補間）
future['出来高'] = future['出来高'].fillna(method='ffill')

# === 予測 ===
forecast = prophet_model.predict(future)

# テスト期間の予測だけ取り出す
forecast_test = forecast.iloc[-len(test_p):]

y_true_p = test_p['y'].values
y_pred_p = forecast_test['yhat'].values

# === 精度評価 ===
rmse_p = np.sqrt(mean_squared_error(y_true_p, y_pred_p))
mae_p = mean_absolute_error(y_true_p, y_pred_p)

print(f"[Prophet] テストデータ RMSE: {rmse_p:.4f}")
print(f"[Prophet] テストデータ MAE: {mae_p:.4f}")

# === プロット ===
plt.figure(figsize=(12,6))
plt.plot(train_p['ds'], train_p['y'], label='Train Actual', color='red')
plt.plot(test_p['ds'], test_p['y'], label='Test Actual', color='orange')
plt.plot(test_p['ds'], y_pred_p, label='Prophet Predicted', color='blue')
plt.title('ProphetによるNTT株価予測（終値 + 出来高）')
plt.xlabel('日付')
plt.ylabel('終値')
plt.legend()
plt.grid(True)
plt.show()

# === トレンド・季節性の分解 ===
prophet_model.plot_components(forecast)
plt.show()

# デバッグ用
print(train_p.head())
print(test_p.head())