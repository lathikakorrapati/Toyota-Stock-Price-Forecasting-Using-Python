import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

#Helper: RMSE 
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Step 1–2: Load & prepare data

df = pd.read_csv("C:/Users/hp/Downloads/archive (2)/Toyota_historical_data.csv")

# Convert Date to datetime with UTC, then drop timezone
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
df["Date"] = df["Date"].dt.tz_convert(None)

# Sort and set index
df = df.sort_values(by="Date")
df = df.set_index("Date")

# Quick info
df.info()
print(df.head())

# Step 3: Basic EDA

# 3.1 Full Close price history
plt.figure(figsize=(12, 4))
plt.plot(df["Close"])
plt.title("Toyota Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.tight_layout()
plt.show()

# 3.2 Last 3 years
last_3_years = df.loc[df.index >= df.index.max() - pd.DateOffset(years=3)]
plt.figure(figsize=(12, 4))
last_3_years["Close"].plot()
plt.title("Toyota Close Price - Last 3 Years")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.tight_layout()
plt.show()
print(df.columns)
print(df["Close"].describe())

# Step 4: Feature Engineering
df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
df["DayOfWeek"] = df.index.dayofweek

# Lag features
df["Close_lag1"] = df["Close"].shift(1)
df["Close_lag7"] = df["Close"].shift(7)
df["Close_lag30"] = df["Close"].shift(30)

# Moving averages
df["MA7"] = df["Close"].rolling(window=7).mean()
df["MA30"] = df["Close"].rolling(window=30).mean()
print(df.head(15))

# Step 5: Train–Test Split
train = df.loc[: "2023-12-31"].copy()
test = df.loc["2024-01-01":].copy()
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.tail())
print(test.head())

# Step 6: Baseline Models

# Naive baseline: yesterday's close as today's forecast
test["Naive_Forecast"] = test["Close"].shift(1)

# 7-day moving average baseline
test["MA7_Forecast"] = test["Close"].rolling(7).mean()

# Drop rows where baselines are NaN
test_eval = test.dropna(subset=["Naive_Forecast", "MA7_Forecast"])
naive_mae = mean_absolute_error(test_eval["Close"], test_eval["Naive_Forecast"])
naive_rmse = rmse(test_eval["Close"], test_eval["Naive_Forecast"])
ma7_mae = mean_absolute_error(test_eval["Close"], test_eval["MA7_Forecast"])
ma7_rmse = rmse(test_eval["Close"], test_eval["MA7_Forecast"])
print("Naive MAE:", naive_mae)
print("Naive RMSE:", naive_rmse)
print("MA7 MAE:", ma7_mae)
print("MA7 RMSE:", ma7_rmse)

# Step 7: ARIMA Model

# 7.1: ADF test on original series
result = adfuller(train["Close"].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# 7.2: ADF test on differenced series
train_diff = train["Close"].diff().dropna()
result_diff = adfuller(train_diff)
print("ADF Statistic after diff:", result_diff[0])
print("p-value after diff:", result_diff[1])

# Use integer index to avoid frequency warnings
train_close = train["Close"].reset_index(drop=True)

# 7.3: Fit ARIMA(1,1,1)
model = ARIMA(train_close, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# 7.4: Forecast into test period
forecast = model_fit.forecast(steps=len(test))
test["ARIMA_Forecast"] = forecast.values  # align with test dates

# 7.5: Evaluate ARIMA
test_arima = test.dropna(subset=["ARIMA_Forecast"])
arima_mae = mean_absolute_error(test_arima["Close"], test_arima["ARIMA_Forecast"])
arima_rmse = rmse(test_arima["Close"], test_arima["ARIMA_Forecast"])
print("ARIMA MAE:", arima_mae)
print("ARIMA RMSE:", arima_rmse)

# 7.6: Plot ARIMA vs Actual
plt.figure(figsize=(12, 5))
plt.plot(test["Close"], label="Actual")
plt.plot(test["ARIMA_Forecast"], label="ARIMA", alpha=0.8)
plt.legend()
plt.title("ARIMA Forecast vs Actual (2024–2025)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.tight_layout()
plt.show()

# Step 8: Prophet Model

# 8.1 Prepare data for Prophet: columns 'ds' (date) and 'y' (target)
df_prophet = df[["Close"]].reset_index()  # Date index → column
df_prophet = df_prophet.rename(columns={"Date": "ds", "Close": "y"})

# Same time split as before
train_prophet = df_prophet[df_prophet["ds"] <= "2023-12-31"].copy()
test_prophet = df_prophet[df_prophet["ds"] >= "2024-01-01"].copy()
print("Prophet train shape:", train_prophet.shape)
print("Prophet test shape:", test_prophet.shape)

# 8.2 Define and fit Prophet model
m = Prophet(
    daily_seasonality=False,   # stock is only on business days
    weekly_seasonality=True,   # weekly pattern
    yearly_seasonality=True    # yearly pattern
)
m.fit(train_prophet)

# 8.3 Create future dataframe and forecast
future = m.make_future_dataframe(periods=len(test_prophet), freq="B")  # 'B' = business days
forecast_p = m.predict(future)

# 8.4 Merge forecast with actual test values (inner join to keep only matching dates)
forecast_small = forecast_p[["ds", "yhat"]]
merged = pd.merge(test_prophet, forecast_small, on="ds", how="inner")

# 8.5 Drop any rows where yhat is NaN (safety)
merged = merged.dropna(subset=["yhat"])

# 8.6 Evaluate Prophet
prophet_mae = mean_absolute_error(merged["y"], merged["yhat"])
prophet_rmse = rmse(merged["y"], merged["yhat"])
print("Prophet MAE:", prophet_mae)
print("Prophet RMSE:", prophet_rmse)

# 8.7 Plot Prophet forecast vs Actual (test period)
plt.figure(figsize=(12, 5))
plt.plot(merged["ds"], merged["y"], label="Actual")
plt.plot(merged["ds"], merged["yhat"], label="Prophet Forecast", alpha=0.8)
plt.title("Prophet Forecast vs Actual (2024–2025)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# 8.8 Plot Prophet components: trend, weekly, yearly
fig_components = m.plot_components(forecast_p)
plt.show()

# Step 9: Model Comparison Table
results = pd.DataFrame(
    [
        {"Model": "Naive", "MAE": naive_mae, "RMSE": naive_rmse},
        {"Model": "MA7", "MAE": ma7_mae, "RMSE": ma7_rmse},
        {"Model": "ARIMA", "MAE": arima_mae, "RMSE": arima_rmse},
        {"Model": "Prophet", "MAE": prophet_mae, "RMSE": prophet_rmse},
    ]
)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================\n")
print(results)
print("\nSorted by MAE (lower is better):")
print(results.sort_values(by="MAE"))

# Step 10–11: Final Visualizations

# 10.1 Full historical Close price (clean version for report)
plt.figure(figsize=(12, 4))
plt.plot(df["Close"])
plt.title("Toyota Close Price - Full History")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.tight_layout()
plt.show()

# 10.2 Train vs Test split (with vertical line)
split_date = pd.to_datetime("2024-01-01")
plt.figure(figsize=(12, 4))
plt.plot(train.index, train["Close"], label="Train")
plt.plot(test.index, test["Close"], label="Test")
plt.axvline(x=split_date, color="k", linestyle="--", label="Train/Test Split")
plt.title("Train vs Test Split (Toyota Close Price)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# 10.3 Actual vs model forecasts on Test period
comparison = pd.DataFrame(index=test.index)
comparison["Actual"] = test["Close"]
comparison["Naive"] = test["Naive_Forecast"]
comparison["MA7"] = test["MA7_Forecast"]
comparison["ARIMA"] = test["ARIMA_Forecast"]

# Prophet forecast (from 'merged')
prophet_series = merged.set_index("ds")["yhat"]
comparison["Prophet"] = prophet_series.reindex(comparison.index)

# Drop rows where we don't have all forecasts
comparison_plot = comparison.dropna(subset=["Actual", "Naive", "ARIMA", "Prophet"])
plt.figure(figsize=(12, 5))
plt.plot(comparison_plot.index, comparison_plot["Actual"], label="Actual", linewidth=2)
plt.plot(comparison_plot.index, comparison_plot["Naive"], label="Naive", alpha=0.8)
plt.plot(comparison_plot.index, comparison_plot["ARIMA"], label="ARIMA", alpha=0.8)
plt.plot(comparison_plot.index, comparison_plot["Prophet"], label="Prophet", alpha=0.8)
# Optional: include MA7 as well
# plt.plot(comparison_plot.index, comparison_plot["MA7"], label="MA7", alpha=0.8)
plt.title("Model Forecasts vs Actual (Test Period: 2024–2025)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# 10.4 Residual plots (Actual - Forecast) for main models
residuals = pd.DataFrame(index=comparison_plot.index)
residuals["Naive"] = comparison_plot["Actual"] - comparison_plot["Naive"]
residuals["ARIMA"] = comparison_plot["Actual"] - comparison_plot["ARIMA"]
residuals["Prophet"] = comparison_plot["Actual"] - comparison_plot["Prophet"]
plt.figure(figsize=(12, 5))
plt.plot(residuals.index, residuals["Naive"], label="Naive Residuals", alpha=0.8)
plt.plot(residuals.index, residuals["ARIMA"], label="ARIMA Residuals", alpha=0.8)
plt.plot(residuals.index, residuals["Prophet"], label="Prophet Residuals", alpha=0.8)
plt.axhline(0, color="k", linestyle="--")
plt.title("Residuals Over Time (Test Period)")
plt.xlabel("Date")
plt.ylabel("Residual (Actual - Forecast)")
plt.legend()
plt.tight_layout()
plt.show()

# 10.5 Residual distribution (histograms)
plt.figure(figsize=(12, 4))
plt.hist(residuals["Naive"].dropna(), bins=30, alpha=0.5, label="Naive")
plt.hist(residuals["ARIMA"].dropna(), bins=30, alpha=0.5, label="ARIMA")
plt.hist(residuals["Prophet"].dropna(), bins=30, alpha=0.5, label="Prophet")
plt.title("Residual Distribution (Test Period)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
