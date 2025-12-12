# VAR Model Streamlit Application
# This app runs your exact VAR forecasting code in a Streamlit interface

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import io
import sys

# Set page config
st.set_page_config(page_title="VAR Model Forecasting", layout="wide")

# Title
st.title("📈 Vector Autoregression (VAR) Model - Stock Price Forecasting")
st.markdown("---")

# Sidebar for file upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

# Sidebar parameters
st.sidebar.header("Model Parameters")
n_obs = st.sidebar.number_input("Number of forecast steps", min_value=10, max_value=500, value=350, step=10)

if uploaded_file is not None:
    try:
        # Load the data
        df_filtered = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
        
        st.success(f"✅ Data loaded successfully! Shape: {df_filtered.shape}")
        
        # Display data info
        with st.expander("📊 View Data Preview"):
            st.dataframe(df_filtered.head(10))
            st.write(f"**Columns:** {list(df_filtered.columns)}")
            st.write(f"**Date Range:** {df_filtered.index.min()} to {df_filtered.index.max()}")
        
        # Check if required columns exist
        required_cols = ['stock_price', 'nasdaq_index', 'sp500_index']
        missing_cols = [col for col in required_cols if col not in df_filtered.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df_filtered.columns)}")
        else:
            # Button to run the model
            if st.button("🚀 Run VAR Model", type="primary"):
                with st.spinner("Running VAR model analysis..."):
                    
                    # Capture console output
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()
                    
                    try:
                        # ============================================================
                        # YOUR EXACT CODE STARTS HERE (UNCHANGED)
                        # ============================================================
                        
                        # --- Use your data with at least two related columns ---
                        # Example: stock_price and sp500_index,nasdaq_index
                        df_var = df_filtered[['stock_price', 'nasdaq_index','sp500_index']].dropna()


                        # --- Difference the data to make it stationary ---
                        df_var_diff = df_var.diff().dropna()


                        # --- Train/Test Split ---
                        # n_obs = 350  # forecast next 30  (commented out - using sidebar parameter)
                        train = df_var_diff[:-n_obs]
                        test = df_var_diff[-n_obs:]


                        # --- Fit VAR model ---
                        model = VAR(train)
                        results = model.fit(ic='aic')  # automatically chooses best lag order
                        print(results.summary())


                        # --- Forecast next 30 steps ---
                        lag_order = results.k_ar
                        forecast_input = train.values[-lag_order:]
                        forecast = results.forecast(y=forecast_input, steps=n_obs)


                        # --- Convert to DataFrame ---
                        forecast_df = pd.DataFrame(forecast, columns=df_var.columns, index=test.index)


                        # --- Reverse differencing to original scale ---
                        last_actual = df_var.iloc[-n_obs - 1]
                        forecast_actual = forecast_df.cumsum() + last_actual


                        # --- Evaluate (only for stock_price) ---
                        actual = df_var.iloc[-n_obs:]['stock_price']
                        predicted = forecast_actual['stock_price']


                        # Mean Squared Error
                        mse = mean_squared_error(actual, predicted)


                        # Root Mean Squared Error
                        rmse = np.sqrt(mse)


                        print(f"\nVAR MODEL MSE: {mse:.3f}")
                        print(f"VAR MODEL RMSE: {rmse:.3f}")


                        # --- Display forecast with timestamps ---
                        print("\nNext 30 Forecasted Stock Prices (with timestamps):")
                        # display(forecast_actual['stock_price'])  # Commented - will show in Streamlit
                        
                        # ============================================================
                        # YOUR EXACT CODE ENDS HERE
                        # ============================================================
                        
                        # Restore stdout
                        sys.stdout = old_stdout
                        console_output = buffer.getvalue()
                        
                        # Display results in Streamlit
                        st.success("✅ Model completed successfully!")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
                        with col2:
                            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")
                        with col3:
                            st.metric("Lag Order Selected", lag_order)
                        
                        st.markdown("---")
                        
                        # Model Summary
                        with st.expander("📋 Model Summary", expanded=True):
                            st.text(console_output)
                        
                        # Forecasted values
                        st.subheader("🔮 Forecasted Stock Prices")
                        st.dataframe(forecast_actual['stock_price'].to_frame().style.format("{:.2f}"))
                        
                        # Download button for forecast
                        csv = forecast_actual['stock_price'].to_csv()
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name="var_forecast.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("---")
                        
                        # Plot 1: Actual Stock Prices
                        st.subheader("📊 Actual Stock Price")
                        fig1, ax1 = plt.subplots(figsize=(12, 6))
                        ax1.plot(df_var['stock_price'], label='Actual Stock Price', color='blue')
                        ax1.set_title('Actual Stock Price', fontsize=16, fontweight='bold')
                        ax1.set_xlabel('Date', fontsize=12)
                        ax1.set_ylabel('Stock Price', fontsize=12)
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig1)
                        
                        # Plot 2: VAR Model Forecast
                        st.subheader("🎯 VAR Model Forecast")
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        ax2.plot(forecast_actual.index, forecast_actual['stock_price'], 
                                label='Forecast (Next 30)', color='red', linestyle='--', linewidth=2)
                        ax2.set_title('VAR Model — 30 Stock Price Forecast', fontsize=16, fontweight='bold')
                        ax2.set_xlabel('Date', fontsize=12)
                        ax2.set_ylabel('Stock Price', fontsize=12)
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                        # Combined Plot
                        st.subheader("📈 Combined View: Actual vs Forecast")
                        fig3, ax3 = plt.subplots(figsize=(14, 7))
                        ax3.plot(df_var.index, df_var['stock_price'], 
                                label='Historical Stock Price', color='blue', linewidth=1.5)
                        ax3.plot(forecast_actual.index, forecast_actual['stock_price'], 
                                label=f'VAR Forecast ({n_obs} steps)', color='red', 
                                linestyle='--', linewidth=2)
                        ax3.axvline(x=forecast_actual.index[0], color='green', 
                                   linestyle=':', linewidth=2, label='Forecast Start')
                        ax3.set_title('Actual Stock Price vs VAR Model Forecast', 
                                     fontsize=16, fontweight='bold')
                        ax3.set_xlabel('Date', fontsize=12)
                        ax3.set_ylabel('Stock Price', fontsize=12)
                        ax3.legend(loc='best')
                        ax3.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        
                        # Additional Analysis
                        st.markdown("---")
                        st.subheader("📉 Forecast Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Actual Stock Price Stats (Test Period)**")
                            st.write(actual.describe())
                        
                        with col2:
                            st.write("**Forecasted Stock Price Stats**")
                            st.write(predicted.describe())
                        
                    except Exception as e:
                        sys.stdout = old_stdout
                        st.error(f"❌ Error running model: {str(e)}")
                        st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Please upload a CSV file to begin")
    
    st.markdown("""
    ### 📝 Requirements:
    - CSV file with datetime index
    - Required columns: `stock_price`, `nasdaq_index`, `sp500_index`
    - Sufficient historical data for training
    
    ### 🎯 What this app does:
    1. Loads your time series data
    2. Creates stationary data using differencing
    3. Fits a VAR (Vector Autoregression) model
    4. Forecasts future stock prices
    5. Evaluates model performance using MSE and RMSE
    6. Visualizes actual vs forecasted values
    
    ### 🚀 How to use:
    1. Upload your CSV file using the sidebar
    2. Adjust forecast parameters if needed
    3. Click "Run VAR Model" button
    4. View results, plots, and download forecasts
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>VAR Model Forecasting Application | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)
