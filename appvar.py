# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained VAR model
try:
    var_model = joblib.load('var_model.joblib')
    models_loaded = True
except FileNotFoundError:
    st.error("Error: VAR model file not found. Please ensure 'var_model.joblib' is in the same directory.")
    models_loaded = False

st.title('Apple Stock Price Forecasting (VAR Model)')

if models_loaded:
    st.write("Forecasting Apple stock prices for the next 30 time steps using the VAR model.")

    # Placeholder for the last known values for VAR model differencing reversal
    # In a real application, you would need to save the last 'lag_order' actual values
    # from your training data to correctly reverse the differencing for the forecast.
    # For this example, we'll use a simplified approach and note the requirement.
    # You need to load this from your training data that was used to train the VAR model.
    # For example, if you saved the last part of your training data as 'df_var_train_last.joblib':
    # try:
    #      last_actual_var = joblib.load('df_var_train_last.joblib')[-var_model.k_ar:]
    # except FileNotFoundError:
    #      st.error("Could not load last actual values for VAR differencing reversal.")
    #      last_actual_var = None

    # --- Simplified approach for demonstration ---
    # Assuming you have access to the original df_var used for training and testing
    # Replace this with the actual last values from your training data if saved separately
    try:
        # This requires the original df_var to be available or saved
        # If df_var is not available, you need to save the last_actual_var during training
        df_var = pd.read_excel("Apples_stock price dataset.xlsx") # Reloading for demonstration
        df_var['timestamp'] = pd.to_datetime(df_var['timestamp'])
        df_var.set_index('timestamp', inplace=True)
        # Filter data like it was done during training
        df_var = df_var[(df_var.index.hour >= 4) & (df_var.index.hour <= 20)]
        df_var = df_var[df_var.index.dayofweek < 5]
        df_var = df_var[['stock_price', 'nasdaq_index','sp500_index']].dropna()

        n_obs = 30 # Same as used in training
        train_size = len(df_var) - n_obs
        train_df_var = df_var.iloc[:train_size]
        last_actual_var = train_df_var.values[-var_model.k_ar:]

    except Exception as e:
        st.error(f"Could not load or process data to get last actual values for VAR differencing reversal: {e}")
        last_actual_var = None


    if st.button('Generate Forecast'):
        if last_actual_var is not None:
            with st.spinner('Generating forecast using VAR model...'):
                try:
                    # Forecast the next 30 steps (in differenced scale)
                    forecast_diff = var_model.forecast(y=last_actual_var, steps=30)
                    forecast_df_diff = pd.DataFrame(forecast_diff, columns=var_model.names)

                    # Reverse differencing to get the forecast in the original scale
                    # Need the last actual value from the training data
                    last_known_value = train_df_var['stock_price'].iloc[-1] # Last stock price from training data
                    forecast_actual_var = forecast_df_diff.cumsum() + last_known_value

                    # For plotting and display, create a time index for the forecast
                    # This assumes hourly data and a continuous sequence after the training data end date
                    last_train_date = train_df_var.index[-1]
                    forecast_index = pd.date_range(start=last_train_date, periods=30 + 1, freq='H')[1:] # Generate hourly index

                    forecast_stock_price = forecast_actual_var['stock_price']
                    forecast_stock_price.index = forecast_index # Assign the generated index

                    st.subheader("VAR Forecast (Next 30 Time Steps)")
                    st.line_chart(forecast_stock_price)
                    st.write(forecast_stock_price)

                except Exception as e:
                    st.error(f"An error occurred during VAR forecasting: {e}")
        else:
             st.warning("VAR forecast cannot be generated due to missing last actual values for differencing reversal.")

else:
    st.write("Please ensure the VAR model file is in the correct directory and rerun the application.")