# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained models
try:
    sarima_model = joblib.load('sarima_model.joblib')
    var_model = joblib.load('var_model.joblib')
    models_loaded = True
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'sarima_model.joblib' and 'var_model.joblib' are in the same directory.")
    models_loaded = False

st.title('Apple Stock Price Forecasting')

if models_loaded:
    st.write("Select a model to forecast Apple stock prices for the next 30 time steps.")

    model_choice = st.selectbox("Choose a Model", ["SARIMA", "VAR"])

    # Placeholder for the last known values for VAR model differencing reversal
    # In a real application, you would need to save the last 'lag_order' actual values
    # from your training data to correctly reverse the differencing for the forecast.
    # For this example, we'll use a simplified approach and note the requirement.
    last_actual_var = None # You need to load this from your training data
    if model_choice == "VAR":
         st.warning("Note: For accurate VAR forecasting, the last 'lag_order' actual values from the training data are needed to reverse differencing.")
         # Example of how you might load the last actual values (replace with your actual data loading/saving)
         # Assuming you saved the original df_var used for training
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
        with st.spinner(f'Generating forecast using {model_choice}...'):
            try:
                if model_choice == "SARIMA":
                    forecast = sarima_model.forecast(steps=30)
                    st.subheader("SARIMA Forecast (Next 30 Time Steps)")
                    st.line_chart(forecast)
                    st.write(forecast)

                elif model_choice == "VAR":
                    if last_actual_var is not None:
                        # This part needs careful implementation based on how you saved the training data
                        # and how you want to handle the index for the forecast.
                        # This is a simplified example.
                        forecast_diff = var_model.forecast(y=last_actual_var, steps=30)
                        forecast_df_diff = pd.DataFrame(forecast_diff, columns=var_model.names)

                        # Simple cumulative sum for reversal - needs adjustment for actual index and initial value
                        # This part is a placeholder and requires proper handling of the last observed value
                        # and the forecast index based on your data's frequency.
                        # A more robust approach would involve saving the last value of the training data.
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
                    else:
                         st.warning("VAR forecast cannot be generated due to missing last actual values for differencing reversal.")


            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")
else:
    st.write("Please ensure the model files are in the correct directory and rerun the application.")