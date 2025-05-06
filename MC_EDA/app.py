import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Streamlit page setup
st.title("SARIMAX Sales Forecasting App")
st.write("""
This app uses the SARIMAX model to forecast sales.
Upload a CSV file containing `dates` and `marketing budget` to generate the forecast for the next period.
""")

# Helper functions
def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the SARIMAX forecast.

    Parameters:
    - y_true (pd.Series): Actual values.
    - y_pred (pd.Series): Predicted values.

    Returns:
    - mse (float): Mean Squared Error.
    - rmse (float): Root Mean Squared Error.
    - mae (float): Mean Absolute Error.
    - r2 (float): R-squared score.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def forecast_with_sarimax(y_train, X_train, forecast_days, marketing_plan):
    """
    Forecast future sales using the SARIMAX model.

    Parameters:
    - y_train (pd.Series): Training target values.
    - X_train (pd.DataFrame): Training feature values.
    - forecast_days (int): Number of days to forecast.
    - marketing_plan (list): Planned marketing budget for the forecast period.

    Returns:
    - forecast_df (pd.DataFrame): Forecasted values with dates as index.
    - fitted_model: Fitted SARIMAX model.
    """
    model = SARIMAX(y_train, exog=X_train[['marketing']], 
                    order=(1, 0, 2), seasonal_order=(2, 1, 2, 30))
    fitted_model = model.fit(disp=False)
    
    # Save the model for future use
    joblib.dump(fitted_model, "sarimax_model.pkl")
    
    # Create future marketing data for forecasting
    future_marketing = pd.DataFrame({'marketing': marketing_plan}, 
                                     index=pd.date_range(start=X_train.index[-1] + pd.Timedelta(days=1), 
                                                         periods=forecast_days))
    forecast = fitted_model.forecast(steps=forecast_days, exog=future_marketing)
    forecast_df = pd.DataFrame({'Forecasted Subtotal': forecast}, index=future_marketing.index)
    
    return forecast_df, fitted_model

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    # Load input data
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(input_data.head())

    # Ensure required columns exist
    if "dates" in input_data.columns and "marketing budget" in input_data.columns:
        input_data["dates"] = pd.to_datetime(input_data["dates"])
        input_data.set_index("dates", inplace=True)
        marketing_plan = input_data["marketing budget"].values
        forecast_days = len(marketing_plan)

        # Load historical data
        historical_data = pd.read_csv("data/daily_sales_data.csv")
        historical_data["Created at"] = pd.to_datetime(historical_data["Created at"])
        historical_data.set_index("Created at", inplace=True)

        features = ["marketing"]
        target = "Subtotal"
        X = historical_data[features]
        y = historical_data[target]

        # Train-test split
        X_train = X[(X.index >= '2022-11-01') & (X.index <= '2025-03-20')]
        X_test = X[X.index > '2025-03-20']
        y_train = y[(X.index >= '2022-11-01') & (X.index <= '2025-03-20')]
        y_test = y[X.index > '2025-03-20']

        # Forecast with SARIMAX
        forecast_df, sarimax_model = forecast_with_sarimax(y_train, X_train, forecast_days, marketing_plan)

        # Evaluate the model
        mse, rmse, mae, r2 = evaluate_model(y_test, forecast_df["Forecasted Subtotal"].reindex(y_test.index))

        # Display results
        st.write(f"### Forecast for the next {forecast_days} days:")
        st.write(forecast_df)

        st.write("### Model Performance Metrics:")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**R²:** {r2:.2f}")

        # Plot the forecast
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="Test Data", color="green")
        plt.plot(forecast_df.index, forecast_df["Forecasted Subtotal"], label="Forecast", color="orange")
        plt.title("SARIMAX Sales Forecast")
        plt.xlabel("Date")
        plt.ylabel("Subtotal")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # def plot_forecast(temp_sales, y_test, forecast_df, title):
        #     # Plot the results
        #     plt.figure(figsize=(12, 6))
        #     plt.subplot(2, 1, 1)
        #     plt.plot(temp_sales.index, temp_sales["Subtotal"], label="Historical Subtotal", color="blue")
        #     plt.subplot(2, 1, 2)
        #     plt.plot(y_test.index, y_test, label="Test Data", color="green")
        #     plt.plot(forecast_df.index, forecast_df["Forecasted Subtotal"], label="4-Week Forecast", color="orange")
        #     plt.title(f"4-Week Forecast {title}")
        #     plt.xlabel("Date")
        #     plt.ylabel("Subtotal")
        #     plt.legend()
        #     plt.grid()
        #     plt.savefig(f'imgs/{title}_forecast.png')
            # plt.show()

        # Save forecast results
        if st.button("Download Forecast Results"):
            forecast_df.to_csv("forecast_results.csv")
            st.write("Forecast results saved as `forecast_results.csv`.")
    else:
        st.error("The uploaded file must contain `dates` and `marketing budget` columns.")