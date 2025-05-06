# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt

# # Function to evaluate model performance
# def evaluate_model(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return mse, rmse, mae, r2

# # Function to train and evaluate a model
# def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler=None):
#     if scaler:
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#     else:
#         X_train_scaled, X_test_scaled = X_train, X_test

#     model.fit(X_train_scaled, y_train)
#     y_train_pred = model.predict(X_train_scaled)
#     y_test_pred = model.predict(X_test_scaled)
#     train_metrics = evaluate_model(y_train, y_train_pred)
#     test_metrics = evaluate_model(y_test, y_test_pred)
#     return model, train_metrics, test_metrics, y_test_pred

# # Function to forecast future values
# def forecast_future(model, X_train, forecast_days, scaler=None):
#     # last_data = X_train.iloc[-1].values.reshape(1, -1)
#     # predictions = []

#     # for _ in range(28):  # Predict for the next 28 days
#     #     # Scale the last data point
#     #     last_data_scaled = scaler.transform(last_data)

#     #     # Make prediction
#     #     next_prediction = model.predict(last_data_scaled)[0]
#     #     predictions.append(next_prediction)
        
#     #     # Update the last data point with the new prediction
#     #     last_data = np.roll(last_data, shift=-1, axis=1)  # Shift features to the left
#     #     last_data[0, -1] = next_prediction  # Add the new prediction to the last feature

#     # # Create a DataFrame for the forecast
#     # forecast_dates = pd.date_range(start=X_train.index[-1] + pd.Timedelta(days=1), periods=28)
#     # forecast_df = pd.DataFrame(predictions, index=forecast_dates, columns=["Forecasted Subtotal"])
#     # return forecast_df
#     last_data = X_train.iloc[-1].copy()
#     future_predictions = []
#     forecast_dates = []
#     marketing_plan = [
#     0, 0, 26254, 30500, 0, 0, 0, 0, 0, 17824, 700, 9407,
#     62584, 0, 54217, 0, 0, 0, 18654, 47874, 0, 0, 0, 0,
#     53430, 0, 0, 65115, 521, 0, 0
# ]

#     for i in range(forecast_days):
#         forecast_date = X_train.index[-1] + pd.Timedelta(days=i + 1)
#         forecast_dates.append(forecast_date)

#         # Update lag features
#         last_data['total_prev_day'] = future_predictions[-1] if future_predictions else last_data['total_prev_day']
#         # last_data['total_prev_week'] = future_predictions[-7] if len(future_predictions) >= 7 else last_data['total_prev_week']
#         last_data['total_same_day_last_week'] = future_predictions[-7] if len(future_predictions) >= 7 else last_data['total_same_day_last_week']

#         # Update rolling statistics
#         # if future_predictions:
#         #     last_7d = future_predictions[-7:] if len(future_predictions) >= 7 else future_predictions
#         #     last_data['total_7d_avg'] = np.mean(last_7d)
#         #     last_data['total_7d_std'] = np.std(last_7d)

#         # Update growth features
#         if len(future_predictions) >= 1:
#             last_data['total_growth_1d'] = ((future_predictions[-1] - future_predictions[-2]) / future_predictions[-2]) if len(future_predictions) >= 2 else 0
#         # if len(future_predictions) >= 7:
#         #     last_data['total_growth_1w'] = ((future_predictions[-1] - future_predictions[-7]) / future_predictions[-7]) if len(future_predictions) >= 7 else 0

#         # Update month and 30-day averages
#         if len(future_predictions) >= 30:
#             last_data['total_prev_month'] = np.mean(future_predictions[-30:])
#         if len(future_predictions) >= 30:
#             last_data['total_30d_avg'] = np.mean(future_predictions[-30:])
#         marketing_value = marketing_plan[i] if i < len(marketing_plan) else marketing_plan[-1]
#         last_data['marketing'] = marketing_value

#         # Scale data if scaler is used
#         last_data_scaled = scaler.transform([last_data.values]) if scaler else [last_data.values]

#         # Make prediction
#         next_prediction = model.predict(last_data_scaled)[0]
#         future_predictions.append(next_prediction)

#     forecast_df = pd.DataFrame({'Forecasted Subtotal': future_predictions}, index=forecast_dates)
#     return forecast_df

# # Function to plot forecasts
# def plot_forecast(temp_sales, y_test, forecast_df, title):
#     # Plot the results
#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(temp_sales.index, temp_sales["Subtotal"], label="Historical Subtotal", color="blue")
#     plt.subplot(2, 1, 2)
#     plt.plot(y_test.index, y_test, label="Test Data", color="green")
#     plt.plot(forecast_df.index, forecast_df["Forecasted Subtotal"], label="4-Week Forecast", color="orange")
#     plt.title("4-Week Forecast using Random Forest")
#     plt.xlabel("Date")
#     plt.ylabel("Subtotal")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Load and preprocess data
# temp_sales = pd.read_csv('data/daily_sales_data.csv')
# temp_sales.dropna(inplace=True)
# temp_sales.set_index('Created at', inplace=True)
# temp_sales.index = pd.to_datetime(temp_sales.index)

# # Define features and target
# #  "total_prev_week","total_growth_1w",
# features = [
#     "total_prev_day","marketing", "total_same_day_last_week",
#  "total_growth_1d", "total_prev_week","total_growth_1w","total_7d_avg", "total_7d_std",
#     "total_prev_month", "total_30d_avg"
# ]
# target = 'Subtotal'

# X = temp_sales[features]
# y = temp_sales[target]

# # Train-test split
# X_train = X[(X.index >= '2022-11-01') & (X.index <= '2025-03-20')]
# X_test = X[X.index > '2025-03-20']
# y_train = y[(X.index >= '2022-11-01') & (X.index <= '2025-03-20')]
# y_test = y[X.index > '2025-03-20']

# scaler = StandardScaler()

# # Initialize models
# models = {
#     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#     "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5),
#     "Linear Regression": LinearRegression()
# }

# # Train and forecast with each model
# forecast_days = 31
# for model_name, model in models.items():
#     trained_model, train_metrics, test_metrics, y_test_pred = train_and_evaluate_model(
#         model, X_train, y_train, X_test, y_test, scaler
#     )
#     forecast_df = forecast_future(trained_model, X_train, forecast_days, scaler)
#     plot_forecast(temp_sales, y_test, forecast_df, model_name)

#     # Display metrics
#     print(f"{model_name} Train Metrics (MSE, RMSE, MAE, R2): {train_metrics}")
#     print(f"{model_name} Test Metrics (MSE, RMSE, MAE, R2): {test_metrics}")




import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# # Function to evaluate model performance
# def evaluate_model(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return mse, rmse, mae, r2

def evaluate_model(forecast_df, y_test):
    """
    Evaluate the performance of a forecast compared to the actual values.

    Parameters:
    - forecast_df (pd.DataFrame): DataFrame containing the forecasted values with the column 'Forecasted Subtotal'.
    - y_test (pd.Series): Series containing the actual test values.

    Returns:
    - mse (float): Mean Squared Error.
    - rmse (float): Root Mean Squared Error.
    - mae (float): Mean Absolute Error.
    - r2 (float): R-squared score.
    """
    # Align indices between forecast and test data
    y_pred = forecast_df["Forecasted Subtotal"].reindex(y_test.index)

    # Ensure alignment
    if y_pred.isnull().any():
        raise ValueError("Forecast and test data indices do not align properly.")

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, mae, r2

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler=None):
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    return model, train_metrics, test_metrics, y_test_pred

# Function to forecast future values using classic models
def forecast_future(model, X_train, forecast_days, scaler=None):
    last_data = X_train.iloc[-1].copy()
    future_predictions = []
    forecast_dates = []
    marketing_plan = [
        0, 0, 26254, 30500, 0, 0, 0, 0, 0, 17824, 700, 9407,
        62584, 0, 54217, 0, 0, 0, 18654, 47874, 0, 0, 0, 0,
        53430, 0, 0, 65115, 521, 0, 0
    ]

    for i in range(forecast_days):
        forecast_date = X_train.index[-1] + pd.Timedelta(days=i + 1)
        forecast_dates.append(forecast_date)

        # Update lag features
        last_data['total_prev_day'] = future_predictions[-1] if future_predictions else last_data['total_prev_day']
        # last_data['total_prev_week'] = future_predictions[-7] if len(future_predictions) >= 7 else last_data['total_prev_week']
        last_data['total_same_day_last_week'] = future_predictions[-7] if len(future_predictions) >= 7 else last_data['total_same_day_last_week']

        # Update rolling statistics
        # if future_predictions:
        #     last_7d = future_predictions[-7:] if len(future_predictions) >= 7 else future_predictions
        #     last_data['total_7d_avg'] = np.mean(last_7d)
        #     last_data['total_7d_std'] = np.std(last_7d)

        # Update growth features
        if len(future_predictions) >= 1:
            last_data['total_growth_1d'] = ((future_predictions[-1] - future_predictions[-2]) / future_predictions[-2]) if len(future_predictions) >= 2 else 0
        # if len(future_predictions) >= 7:
        #     last_data['total_growth_1w'] = ((future_predictions[-1] - future_predictions[-7]) / future_predictions[-7]) if len(future_predictions) >= 7 else 0

        # Use the marketing plan for the forecast period
        marketing_value = marketing_plan[i] if i < len(marketing_plan) else marketing_plan[-1]
        last_data['marketing'] = marketing_value

        # Scale data if scaler is used
        last_data_scaled = scaler.transform([last_data.values]) if scaler else [last_data.values]

        # Make prediction
        next_prediction = model.predict(last_data_scaled)[0]
        future_predictions.append(next_prediction)

    forecast_df = pd.DataFrame({'Forecasted Subtotal': future_predictions}, index=forecast_dates)
    return forecast_df

# Function to forecast using Prophet
def forecast_with_prophet(X_train,y_train,X_test, forecast_days, marketing_plan):
    X_train.reset_index(inplace=True)
    df_prophet = X_train[['Created at', 'marketing'] ].rename(
        columns={'Created at': 'ds'}
    )
    df_prophet['y'] = y_train[0]
    # Convert the 'ds' column to timezone-naive datetime format
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_convert(None)
    df_prophet.dropna(inplace=True)
    print(df_prophet.isnull().sum())

    model = Prophet()
    model.add_regressor('marketing')  # Add 'marketing' as a regressor
    model.fit(df_prophet)
    future_dates = pd.date_range(start=X_test, periods=len(marketing_plan), freq='D')
    future = pd.DataFrame({'ds': future_dates})
    future['marketing'] = marketing_plan  
    forecast = model.predict(future)
    # Return the forecasted 'yhat' values

    return forecast[['ds', 'yhat']].set_index('ds')



# Function to forecast using ARIMA
def forecast_with_arima(y_train,X_train, forecast_days):
    model = ARIMA(y_train, order=(1, 0, 2))  
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=X_train.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Forecasted Subtotal': forecast}, index=forecast_index)

# Function to forecast using SARIMAX (1, 0, 2, 2, 1, 2, 30)
def forecast_with_sarimax(y_train,X_train, forecast_days, marketing_plan):
    model = SARIMAX(y_train, exog=X_train[['marketing']], order=(1, 0, 2), seasonal_order=(2, 1, 2, 30))
    fitted_model = model.fit(disp=False)
    joblib.dump(fitted_model, 'models/sarimax_model.pkl')
    future_marketing = pd.DataFrame({'marketing': marketing_plan}, index=pd.date_range(start=temp_sales.index[-1] + pd.Timedelta(days=1), periods=forecast_days))
    forecast = fitted_model.forecast(steps=forecast_days, exog=future_marketing)
    forecast_index = pd.date_range(start=X_train.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Forecasted Subtotal': forecast}, index=forecast_index)

def sarimax_grid_search(y_train, exog_train, y_test, exog_test, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    # Define all possible parameter combinations
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

# Function to plot forecasts
def plot_forecast(temp_sales, y_test, forecast_df, title):
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(temp_sales.index, temp_sales["Subtotal"], label="Historical Subtotal", color="blue")
    plt.subplot(2, 1, 2)
    plt.plot(y_test.index, y_test, label="Test Data", color="green")
    plt.plot(forecast_df.index, forecast_df["Forecasted Subtotal"], label="4-Week Forecast", color="orange")
    plt.title(f"4-Week Forecast {title}")
    plt.xlabel("Date")
    plt.ylabel("Subtotal")
    plt.legend()
    plt.grid()
    plt.savefig(f'imgs/{title}_forecast.png')
    # plt.show()

# Load and preprocess data
temp_sales = pd.read_csv('data/daily_sales_data.csv')
temp_sales.dropna(inplace=True)
temp_sales.set_index('Created at', inplace=True)
temp_sales.index = pd.to_datetime(temp_sales.index)

# Define features and target
# "total_growth_1w", "total_7d_avg", "total_7d_std", "total_prev_week", 
features = [
    "total_prev_day", "marketing", "total_same_day_last_week", "total_growth_1d",
    "total_prev_month", "total_30d_avg"
]
target = 'Subtotal'

X = temp_sales[features]
y = temp_sales[target]

# Train-test split
X_train = X[(X.index >= '2022-11-01') & (X.index <= '2025-03-20')]
X_test = X[X.index > '2025-03-20']
y_train = y[(X.index >= '2022-11-01') & (X.index <= '2025-03-20')]
y_test = y[X.index > '2025-03-20']

scaler = StandardScaler()
joblib.dump(scaler, f"models/scaler.pkl")
model_rmse = {}
# Train and forecast with each model
forecast_days = 31
marketing_plan = [
    0, 0, 26254, 30500, 0, 0, 0, 0, 0, 17824, 700, 9407,
    62584, 0, 54217, 0, 0, 0, 18654, 47874, 0, 0, 0, 0,
    53430, 0, 0, 65115, 521, 0, 0
]

# Classic models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100,max_depth=None, random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5),
    "Linear Regression": LinearRegression(),
    "lightgbm": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,random_state=42),
    "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5, verbose=0,random_state=42),
    "ridge": Ridge(alpha=1.0, max_iter=1000, tol=0.001, solver='auto',random_state=42),
    "lasso": Lasso(alpha=0.1, max_iter=1000, tol=0.001,random_state=42),
}
# Train and forecast with classic models
for model_name, model in models.items():
    # Train and evaluate
    trained_model, train_metrics, test_metrics, y_test_pred = train_and_evaluate_model(
        model, X_train, y_train, X_test, y_test, scaler
    )
    forecast_df = forecast_future(trained_model, X_train, forecast_days, scaler)

    # Save the trained model and scaler
    joblib.dump(trained_model, f"models/{model_name}_model.pkl")


    # Calculate RMSE
    rmse = test_metrics[1]  # RMSE is the second value returned by evaluate_model
    model_rmse[model_name] = rmse

    plot_forecast(temp_sales, y_test, forecast_df, model_name)

# # Prophet
# forecast_df = forecast_with_prophet(X_train,y_train,X_test, forecast_days, marketing_plan)
# prophet_rmse = np.sqrt(mean_squared_error(y_test, forecast_df['yhat'].iloc[:len(y_test)]))
# model_rmse["Prophet"] = prophet_rmse
# plot_forecast(temp_sales, y_test, forecast_df, "Prophet")

# ARIMA
forecast_df = forecast_with_arima(y_train, X_train, forecast_days)
arima_rmse = np.sqrt(mean_squared_error(y_test, forecast_df['Forecasted Subtotal'].iloc[:len(y_test)]))
model_rmse["ARIMA"] = arima_rmse
plot_forecast(temp_sales, y_test, forecast_df, "ARIMA")



exog_train = X_train[['marketing']]  # Exogenous variable
exog_test = X_test[['marketing']]

# Define SARIMAX parameter ranges for grid search
p_range = range(0, 3)  # AR terms
d_range = range(0, 2)  # Differencing terms
q_range = range(0, 3)  # MA terms
P_range = range(0, 3)  # Seasonal AR terms
D_range = range(0, 2)  # Seasonal differencing terms
Q_range = range(0, 3)  # Seasonal MA terms
s = 30 # Seasonal period (e.g., 7 for weekly seasonality)

# Perform SARIMAX grid search
# best_model, best_params, best_rmse = sarimax_grid_search(
#     y_train, exog_train, y_test, exog_test,
#     p_range, d_range, q_range, P_range, D_range, Q_range, s
# # )
# print(f"Best SARIMAX parameters: {best_params}")

# SARIMAX
forecast_df = forecast_with_sarimax(y_train, X_train, forecast_days, marketing_plan)
sarimax_rmse = np.sqrt(mean_squared_error(y_test, forecast_df['Forecasted Subtotal'].iloc[:len(y_test)]))
model_rmse["SARIMAX"] = sarimax_rmse
plot_forecast(temp_sales, y_test, forecast_df, "SARIMAX")




# Save the RMSE comparison graph
plt.figure(figsize=(10, 6))
plt.bar(model_rmse.keys(), model_rmse.values(), color='skyblue', alpha=0.7)
plt.ylabel('RMSE')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('imgs/model_rmse_comparison.png')
plt.show()