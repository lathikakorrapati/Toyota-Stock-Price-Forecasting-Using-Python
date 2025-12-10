📈 Toyota Stock Price Forecasting using Time-Series Models
This project performs a full end-to-end time-series forecasting analysis using historical stock data of Toyota Motor Corporation. The objective is to analyze historical price behavior and evaluate multiple forecasting models to predict future closing prices.

🔍 Project Overview
This analysis explores Toyota’s daily stock prices from 1980 to 2025.
The workflow includes:

Data loading and preparation

Date indexing and transformation

Exploratory time-series analysis

Feature engineering (lags & moving averages)

Train–test split

Baseline models

ARIMA forecasting

Prophet forecasting

Performance comparison

Final visualization of results

🧠 Forecasting Models Implemented
Model	Description
Naive Forecast	Uses previous day value
7-Day Moving Average	Rolling average baseline
ARIMA	Autoregressive Integrated Moving Average
Prophet	Additive model with trend & seasonality

📊 Key Visualizations
The project generates the following analytical charts:

Full historical closing prices

Last-3-year price view

ARIMA vs Actual

Prophet vs Actual

Train vs Test split

Model comparison chart

Residual plots

Prophet trend and seasonality components

📈 Evaluation Metrics
Each model is evaluated using:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

This allows objective comparison of traditional and advanced forecasting methods.

🔎 Key Insights
Stock prices demonstrate visible long-term growth and fluctuation cycles

Naive & MA baselines perform reasonably for short-range forecasts

ARIMA and Prophet reveal trend patterns and seasonal behaviour

Performance differs based on horizon and volatility

🧩 Skills Demonstrated
Time-Series Analysis

Forecast Modelling

Python data analysis (Pandas, NumPy)

ARIMA & Prophet models

Visualization using Matplotlib

Statistical evaluation

📦 Technologies Used
Python

Pandas

NumPy

Matplotlib

Statsmodels

Prophet

🚀 Possible Extensions
LSTM / Deep Learning forecasting

Auto-ARIMA or hyperparameter tuning

Prediction intervals

Volatility forecasting

Financial dashboard (Power BI)

