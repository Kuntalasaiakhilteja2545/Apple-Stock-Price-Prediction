📈 Apple Stock Price Forecasting (AAPL)

Machine Learning • Time-Series Analysis • Financial Modelling • Streamlit App

This project focuses on forecasting Apple (AAPL) stock prices for the next 30 days using a combination of machine learning and advanced time-series models. It integrates macroeconomic indicators, market indices, and real trading patterns to create a realistic and data-driven prediction system.

🚀 Project Overview

Using Python, I built an end-to-end forecasting solution that includes data collection, cleaning, feature engineering, model training, evaluation, and deployment. The final output is an interactive Streamlit web app that allows users to visualize stock trends and future predictions in real time.

🧠 What I Did (Summary)

Collected historical AAPL data and combined it with NASDAQ, S&P 500, interest rates, inflation, unemployment, and market sentiment

Preprocessed the data using Pandas/NumPy, including missing value imputation and outlier correction

Filtered data based on active U.S. market hours to match real trading behavior

Experimented with multiple forecasting models:

ARIMA, SARIMA

LSTM (TensorFlow/Keras)

Prophet

Vector Auto Regression (VAR) – Best performing model

Visualized trends, correlations, and predictions using Matplotlib

Deployed a fully interactive Streamlit dashboard for user-driven predictions

🔍 Why VAR Worked Best

The VAR model captured relationships between multiple financial indicators, allowing the forecast to respond to broader market conditions rather than relying solely on past AAPL prices.

🛠 Tech Stack

Python, Pandas, NumPy, Scikit-learn, Statsmodels, TensorFlow/Keras, Prophet, Matplotlib, Streamlit

📊 Features of the Streamlit App

Real-time forecast visualization

30-day price predictions

User interaction with model inputs

Clean and simple UI for investors and beginners

📂 Project Workflow

EDA → Data Cleaning → Feature Engineering → Model Building → Evaluation → Deployment

📜 Project Goal

To build a reliable, data-driven forecasting system that supports investor decision-making through accurate and interactive stock predictions.
