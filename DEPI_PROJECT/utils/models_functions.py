import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import mlflow
import warnings
warnings.filterwarnings('ignore')
import joblib
import os


def sarimax_grid_search(y_train, y_test, p_range, d_range, q_range, P_range, D_range, Q_range, s, exog_train = None , exog_test = None):
    """
    Perform grid search for SARIMAX model parameters.
    Args:
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Test target variable.
        p_range (list): List of p values to test.
        d_range (list): List of d values to test.
        q_range (list): List of q values to test.
        P_range (list): List of P values to test.
        D_range (list): List of D values to test.
        Q_range (list): List of Q values to test.
        s (int): Seasonal period.
        exog_train (pd.DataFrame): Training exogenous variables.
        exog_test (pd.DataFrame): Test exogenous variables.
    Returns:
        best_model (SARIMAX): Best fitted SARIMAX model.
        best_params (tuple): Best parameters (p, d, q, P, D, Q, s).
        best_rmse (float): Best RMSE value.
    """
    import itertools
    param_combinations = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range, [s]))
    best_rmse = float('inf')
    best_params = None
    best_model = None

    print(f"Testing {len(param_combinations)} parameter combinations...")

    for params in param_combinations:
        try:
            # Unpack parameters
            p, d, q, P, D, Q, s = params

            # Define and fit SARIMAX model
            model = SARIMAX(
                y_train, 
                exog=exog_train, 
                order=(p, d, q), 
                seasonal_order=(P, D, Q, s), 
                enforce_stationarity=False, 
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

            # Forecast on the test set
            forecast = model_fit.forecast(steps=len(y_test), exog=exog_test)

            # Calculate RMSE
            rmse = mean_absolute_error(y_test, forecast)
            print(f"Parameters: {params}, RMSE: {rmse:.2f}")

            # Update best model if RMSE improves
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model_fit
        except Exception as e:
            print(f"Error for parameters {params}: {e}")
            continue

    print(f"\nBest Parameters: {best_params}, Best RMSE: {best_rmse:.2f}")
    return best_model, best_params, best_rmse


# Function to forecast using SARIMAX (1, 0, 2, 2, 1, 2, 30)
def train_sarimax_sales_forecasting(df, marketing_plan):
    """
    Train SARIMAX model for sales forecasting.
    Args:
        y_train (pd.Series): Training target variable.
        X_train (pd.DataFrame): Training exogenous variables.
        forecast_days (int): Number of days to forecast.
        marketing_plan (list): List of marketing values for the forecast period.
    Returns:
        None
    """
    print("Executed sarimax sales forecasting train")
    with mlflow.start_run():
        y_train = df[["Created at","Subtotal"]]
        # Ensure the column exists and then type cast it to float
        # Remove commas and convert to float
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].str.replace(",", "").astype(float)
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].replace([np.inf, -np.inf], np.nan)
        marketing_plan["marketing budget"] = marketing_plan["marketing budget"].fillna(method='ffill')  # or 'bfill', or use a fixed value

        print(marketing_plan["marketing budget"].head())

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model = SARIMAX(y_train, exog=marketing_plan["marketing budget"], order=(1, 0, 2), seasonal_order=(2, 1, 2, 30))
        fitted_model = model.fit(disp=False)
        rmse = np.sqrt(mean_squared_error(y_train, fitted_model.fittedvalues))
        r2 = r2_score(y_train, fitted_model.fittedvalues)
        mse = mean_squared_error(y_train, fitted_model.fittedvalues)
        mae = mean_absolute_error(y_train, fitted_model.fittedvalues)
        mape = np.mean(np.abs((y_train - fitted_model.fittedvalues) / y_train)) * 100

        

        joblib.dump(fitted_model, f'models/sales_forecasting_models/sarimax_forecasting_model_{timestamp}.pkl')
        mlflow.log_artifact(f'models/sales_forecasting_models/sarimax_forecasting_model_{timestamp}.pkl', artifact_path='models')
        
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("SARIMAX_order", (1, 0, 2))
        mlflow.log_param("SARIMAX_seasonal_order", (2, 1, 2, 30))
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAPE", mape)

    
        
        print("Pkl model saved as sarimax_model.pkl")


def forecast_sales(start_day,forecast_days, marketing_plan):
    """
    Forecast sales using a pre-trained SARIMAX model.
    Args:
        start_day (str): Start date for the forecast in 'YYYY-MM-DD' format.
        forecast_days (int): Number of days to forecast.
        marketing_plan (list): List of marketing values for the forecast period.
    Returns:
        pd.DataFrame: DataFrame containing the forecasted sales.
    """
    model_dir = 'models/sales_forecasting_models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    latest_model_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = os.path.join(model_dir, latest_model_file)
    print(f"Loading model from {model_path}")

    model = joblib.load(model_path)

    future_marketing = pd.DataFrame({'marketing': marketing_plan}, index=pd.date_range(start=start_day , periods=forecast_days))
    forecast = model.forecast(steps=forecast_days, exog=future_marketing)
    # # Generate the forecast index
    forecast_index = pd.date_range(start=start_day, periods=forecast_days)

    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Subtotal': forecast})
    forecast_df.set_index('Date', inplace=True)

    return forecast_df





def train_sarimax_demand_forecasting(df):
    """
    Forecast item demand using SARIMAX model.
    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Lineitem quantity' and 'category' columns.
    Returns:
        None
    """
    df["Created at"] = pd.to_datetime(df["Created at"], utc=True)
    df.set_index("Created at",inplace =True)
    # print(df.head())
    items = df['category'].unique()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    directory = f'models/demand_categories_models_{timestamp}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for item in items:
        with mlflow.start_run(nested=True): 

            # Filter for a specific item (e.g., 'towels')
            df_item = df[df['category'] == item]

    
            daily_demand = df_item['Lineitem quantity'].resample('D').sum().fillna(0)

            train = daily_demand

            # Best Parameters: (1, 0, 1, 1, 0, 1, 7), Best RMSE: 17.96
            model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 30))  # Weekly seasonality
            results = model.fit(disp=False)
            rmse = np.sqrt(mean_squared_error(train, results.fittedvalues))
            # Save the model
            model_path = f'models/demand_categories_models_{timestamp}/sarimax_model_{item}.pkl'
            joblib.dump(results, model_path)

            mlflow.log_artifact(model_path, artifact_path='models')
            mlflow.log_param("item", item)
            mlflow.log_param("timestamp", timestamp)
            mlflow.log_param("SARIMAX_order", (1, 0, 1))
            mlflow.log_param("SARIMAX_seasonal_order", (1, 0, 1, 30))
            mlflow.log_metric("RMSE", rmse)




def forecast_demand(start_date,forecast_days, item):
    """
    Forecast item demand using a pre-trained SARIMAX model.

    Args:
        start_date (datetime.date): Start date for forecast.
        forecast_days (int): Number of days to forecast.
        item (str): Item category to forecast.

    Returns:
        item_forecasted_demand (float): Total forecasted demand.
        forecast_df (pd.DataFrame): Forecasted demand with dates.
    """
    base_dir = 'models'
    model_dirs = [d for d in os.listdir(base_dir) if d.startswith('demand_categories_models_')]
    if not model_dirs:
        raise ValueError("No model directories found.")

    # Get latest directory based on modification time
    latest_model_dir = max(model_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    latest_model_path = os.path.join(base_dir, latest_model_dir)

    # Find and load the specific item model
    item_model_path = os.path.join(latest_model_path, f'sarimax_model_{item}.pkl')


    try:
        model = joblib.load(item_model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Forecast
    forecast = model.forecast(steps=forecast_days)
    forecast = np.ceil(forecast).clip(lower=0)


    item_forecasted_demand = forecast.sum()

    forecast_df = pd.DataFrame(forecast)
    print(forecast_df)

    return item_forecasted_demand, forecast_df



