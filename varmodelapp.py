import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

st.title("Stock Price Forecasting with VAR Model")

st.write("""
This application forecasts the next 300 stock prices using a Vector Autoregression (VAR) model trained on historical data including stock price, Nasdaq index, and S&P 500 index.
""")

# --- Data Loading and Preparation (Simulated based on notebook context) ---
# In a real application, you would load your data from a file or database.
# For this example, we'll assume df_filtered (from your notebook) is available or recreated.
# We'll simulate loading the data and performing necessary preprocessing steps
# based on your notebook's logic (filtering trading hours, outlier capping).

# ***Important Note:*** In a production app, you should load preprocessed data
# or the saved trained model instead of reprocessing/refitting every time.
# This simulation is for demonstration based on your notebook's flow.

@st.cache_data # Cache the data loading and preprocessing for performance
def load_and_preprocess_data():
    try:
        # Assuming the original data is available or can be loaded
        # Replace with your actual data loading path/method
        # For demonstration, we'll create dummy data with similar structure and scale
        # In a real app, replace this with loading your actual dataset
        data = {
            'timestamp': pd.date_range(start='2010-01-01', periods=100000, freq='h'),
            'stock_price': np.random.rand(100000) * 400 + 100, # Simulate stock prices
            'nasdaq_index': np.random.rand(100000) * 500000 + 8000, # Simulate Nasdaq
            'sp500_index': np.random.rand(100000) * 200000 + 3000,  # Simulate S&P 500
            'inflation_rate': np.random.rand(100000) * 4 + 1,
            'unemployment_rate': np.random.rand(100000) * 4 + 3,
            'interest_rate': np.random.rand(100000) * 3 + 0.5,
            'market_sentiment': np.random.rand(100000) * 2 - 1
        }
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Simulate filling missing values and capping outliers as in the notebook
        df_filled = df.ffill() # Forward fill missing values (as in notebook)

        # Simulate outlier capping (using the function from notebook)
        # In a real app, apply your actual capping logic/values
        def cap_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            return df

        numerical_cols = ['stock_price', 'nasdaq_index', 'sp500_index',
                          'inflation_rate', 'unemployment_rate', 'interest_rate', 'market_sentiment']
        for col in numerical_cols:
             df_filled = cap_outliers_iqr(df_filled, col)


        # Simulate filtering by trading hours and excluding weekends
        df_filtered = df_filled[(df_filled.index.hour >= 4) & (df_filled.index.hour <= 20)]
        df_filtered = df_filtered[df_filtered.index.dayofweek < 5]

        return df_filtered

    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        return None

df_filtered = load_and_preprocess_data()

if df_filtered is not None:
    # Select columns for VAR model
    df_var = df_filtered[['stock_price', 'nasdaq_index', 'sp500_index']].copy()

    # --- Train VAR Model ---
    # In a real application, you would load a pre-trained model.
    # Fitting here for demonstration.
    @st.cache_resource # Cache the model fitting
    def fit_var_model(data):
        try:
            # Difference the data to make it stationary
            data_diff = data.diff().dropna()

            # Fit VAR model (automatically chooses best lag order)
            # Using the full differenced data for fitting in this app simulation
            model = VAR(data_diff)
            results = model.fit(ic='aic')
            return results
        except Exception as e:
            st.error(f"Error fitting VAR model: {e}")
            return None

    var_results = fit_var_model(df_var)

    if var_results is not None:
        st.write(f"VAR Model fitted with lag order: {var_results.k_ar}")

        # --- Generate Forecast ---
        forecast_steps = 300 # Predict next 300 steps

        try:
            # Prepare input for forecasting: the last lag_order observations of the differenced data
            lag_order = var_results.k_ar
            # Ensure we have enough data points for the input
            if len(df_var) < lag_order + 1:
                st.error(f"Not enough data points ({len(df_var)}) to create forecast input with lag order {lag_order}. Need at least {lag_order + 1}.")
            else:
                # Difference the required last observations
                forecast_input = df_var.tail(lag_order + 1).diff().dropna().values[-lag_order:]

                # Generate forecast in differenced values
                forecast_diff = var_results.forecast(y=forecast_input, steps=forecast_steps)

                # Convert to DataFrame with appropriate index
                # The index should start after the last timestamp in the original df_var
                last_timestamp = df_var.index[-1]
                forecast_index = pd.date_range(start=last_timestamp, periods=forecast_steps + 1, freq='h')[1:]
                forecast_df_diff = pd.DataFrame(forecast_diff, columns=df_var.columns, index=forecast_index)

                # Reverse differencing to get forecast in original scale
                # Need the last actual observation in original scale to reverse the differencing
                last_actual_observation = df_var.iloc[-1]
                forecast_actual = forecast_df_diff.cumsum() + last_actual_observation

                st.subheader("Forecasted Stock Prices (Next 300 steps):")
                st.dataframe(forecast_actual['stock_price'])

                # --- Visualize Forecast ---
                st.subheader("Stock Price Forecast Plot:")

                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot recent historical data
                # Show a reasonable amount of recent history for context (e.g., last 300 data points before forecast)
                recent_history = df_var.tail(300)
                ax.plot(recent_history.index, recent_history['stock_price'], label='Recent Actual Prices', color='blue')

                # Plot the forecast
                ax.plot(forecast_actual.index, forecast_actual['stock_price'], label=f'VAR Forecast ({forecast_steps} steps)', color='red', linestyle='--')

                ax.set_title('Stock Price Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Stock Price')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error generating or displaying forecast: {e}")

    else:
        st.error("VAR model could not be fitted.")
else:
    st.error("Data could not be loaded or preprocessed.")
