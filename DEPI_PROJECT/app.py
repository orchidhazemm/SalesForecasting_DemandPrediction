# app.py
from utils.models_functions import *
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from utils.preprocessing import * 
from utils.EDA import * 
from datetime import date
# from models import *
import os

# Page configuration
st.set_page_config(
    page_title="Sales Forcasting & ON-Demand Predictions",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Forecasting Sales", "Demand Product", "Visualizations", "Retraining models"])

# Forecasting Sales Page
def forecasting_sales_page():
    st.title("📈 Sales Forecasting")
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     start_date = st.date_input("Start Date", min_value=datetime.today())
    # with col2:
    #     end_date = st.date_input("End Date", min_value=start_date)
    
    marketing_budget = st.file_uploader("Upload CSV file that contains Marketing budgert & Dates", type=['csv'])
    
    if marketing_budget is not None:
        try:
            marketing_budget_csv = pd.read_csv(marketing_budget)
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(marketing_budget_csv.head())
            st.write("Columns in the dataset:")

            start_date = marketing_budget_csv['dates'].min()
            end_date = marketing_budget_csv['dates'].max()

            st.write(f"Start Date: {start_date}")
            st.write(f"End Date: {end_date}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    
    if st.button("Generate Forecast"):
        # if start_date > end_date:
        #     st.error("Start date must be before end date!")
        if 1:
            days = len(marketing_budget_csv['dates'])
            forecast_df = forecast_sales(start_date,days, marketing_budget_csv['marketing budget'].tolist())
            
            st.line_chart(forecast_df['Forecasted Subtotal'])
            
            st.subheader("Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Daily Sales", f"${forecast_df['Forecasted Subtotal'].mean():,.2f}")
            with col2:
                st.metric("Total Sales", f"${forecast_df['Forecasted Subtotal'].sum():,.2f}")
            with col3:
                st.metric("Peak Sales", f"${forecast_df['Forecasted Subtotal'].max():,.2f}")

# Demand Product Page
def demand_product_page():
    st.title("📊 Demand Product Analysis")

    item = st.selectbox(
        "Select a product category:",
        ['hand towel', 'towel', 'face towel', 'bath mat', 'mattress protector',
         'mattress topper', 'bathrobe', 'pillow', 'coverlet', 'bed sheet',
         'fitted sheet', 'kitchen towel', 'cushion', 'duvet', 'blanket',
         'beach towel', 'set', 'apron', 'slipper', 'other']
    )

    start_date = st.date_input("Start Date", min_value=date(2025, 1, 17))

    if st.button("Process Demand Analysis"):
        try:
            item_forecasted_demand, predictions_df = forecast_demand(start_date, 30, item)

            if predictions_df.empty or predictions_df['predicted_mean'].isna().all():
                st.warning("Forecast data is empty or contains only NaNs. Please check the model or input.")
                return

            st.subheader("📈 Predicted Demand")
            st.dataframe(predictions_df.head())
            st.line_chart(predictions_df['predicted_mean'])

            st.subheader("📌 Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_demand = round(predictions_df['predicted_mean'].mean(), 0
                                    )
                st.metric("Average Daily Demand", f"{avg_demand}")
            with col2:
                total_demand = int(item_forecasted_demand)
                st.metric("Total Demand", f"{total_demand}")
            with col3:
                peak_demand = round(predictions_df['predicted_mean'].max(), 2)
                st.metric("Peak Demand", f"{peak_demand}")

        except ValueError as ve:
            st.error(f"Value Error: {ve}")
        except FileNotFoundError:
            st.error("Model file not found. Please ensure models are trained and stored correctly.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
                


# Dashboards Page
def dashboards_page():
    st.title("📊 Data Visualization Dashboard")
    
    # Top-level selection
    st.subheader("Choose Analysis Type:")
    analysis_type = st.radio("Choose Analysis Type", ["General Graphs", "Specified Analysis"], label_visibility="collapsed")

    # Load data once
    df = pd.read_csv("data/depi_grouped.csv")
    df_ungrouped = pd.read_csv("data/depi_ungrouped.csv")
    df1, df2 = split_numeric_categorical(df)  # or your own specified DataFrames
    df1_ungrouped, df2_ungrouped = split_numeric_categorical(df_ungrouped)  # or your own specified DataFrames
    df_time_series = pd.read_csv("data/depi_time_series.csv")

    if analysis_type == "General Graphs":
        if df is not None:
            try:
                columns = df.columns.tolist()
                numeric_columns = df1.columns.tolist()  # df1 assumed numeric

                plot_columns = ['Select plot type', 'Bar Plot', 'Line Plot', 'Pie Chart', 'Box Plot', 'Histogram', 'Scatter Plot']
                plot_type = st.selectbox("Select plot type", options=plot_columns)

                if plot_type != 'Select plot type':
                    if plot_type in ['Pie Chart', 'Box Plot', 'Histogram']:
                        if plot_type == 'Box Plot':
                            col1 = st.selectbox("Select numeric column", options=numeric_columns)
                        else:
                            col1 = st.selectbox("Select column", options=columns)
                    elif plot_type in ['Bar Plot', 'Line Plot', 'Scatter Plot']:
                        col1 = st.selectbox("Select first column for comparison", options=columns)
                        col2 = st.selectbox("Select second column for comparison", options=columns)

                    match plot_type:
                        case 'Bar Plot':
                            st.subheader(f"Bar Plot of {col1} vs {col2}")
                            bar_plot(df_ungrouped, col1, col2)
                        case 'Line Plot':
                            st.subheader(f"Line Plot of {col1} vs {col2}")
                            line_plot(df, col1, col2)
                        case 'Pie Chart':   
                            st.subheader(f"Pie Chart of {col1}")
                            pie_chart(df, col1)
                        case 'Box Plot':
                            st.subheader(f"Box Plot of {col1}")
                            box_plot(df, col1)
                        case 'Histogram':   
                            st.subheader(f"Histogram of {col1}")
                            histogram(df, col1)
                        case 'Scatter Plot':    
                            st.subheader(f"Scatter Plot of {col1} vs {col2}")
                            scatter_plot(df, col1, col2)

            except Exception as e:
                st.error(f"Error creating visualizations: {str(e)}")

    elif analysis_type == "Specified Analysis":
        try:
            graphs = [
                'Select Analytics',
            "Top 20 Best-selling products",
            "Order Size & Value Analysis",
            "Seasonal Revenue Patterns by Year",
            "seasonal Patterns"]

            graph = st.selectbox("Select Analytics", options=graphs)
            if graph != 'Select Analytics':
                match graph:
                    case 'Top 20 Best-selling products':
                        st.subheader("Top 20 Best-selling Products")
                        top_20_BSP(df_ungrouped)
                    case 'Order Size & Value Analysis':
                        st.subheader("Order Size & Value Analysis")
                        order_analysis(df)
                    case 'Seasonal Revenue Patterns by Year':
                        st.subheader("Seasonal Revenue Patterns by Year")
                        seasonal_Revenue(df)
                    case 'seasonal Patterns':
                        st.subheader("Seasonal Patterns")
                        seasonal(df)
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")

def re_training_models():
    st.title("📊 RE-Training Models")
    st.subheader("Upload CSV with the new data")
    uploaded_new_data_file = st.file_uploader("Upload CSV file", type=['csv'], key="new_data")

    st.subheader("Upload CSV with the new marketing data")
    uploaded_new_marketing_file = st.file_uploader("Upload CSV file", type=['csv'], key="new_marketing")

    
    
    
    if (uploaded_new_data_file and uploaded_new_marketing_file):
        # try:
        df = pd.read_csv(uploaded_new_data_file)
        df_marketing = pd.read_csv(uploaded_new_marketing_file)
        st.success("File uploaded successfully!")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.subheader("Marketing Data Preview")
        st.dataframe(df_marketing.head())
        # st.dataframe(investegating_data(df))

        # Directory where the Depi files are stored
        directory = "data/"
        # Pattern to match the versioned files
        pattern = r"depi_v(\d+)\.csv"
        version_numbers = []

        # List all files in the directory and find the highest version number
        for filename in os.listdir(directory):
            match = re.match(pattern, filename)
            if match:
                version_numbers.append(int(match.group(1)))

        # Determine the highest version number
        if version_numbers:
            highest_version = max(version_numbers)
            highest_version_file = f"depi_v{highest_version}.csv"
        else:
            highest_version = 0
            highest_version_file = "depi_v0.csv"  # Default if no files exist

        # Load the highest version file
        df2 = pd.read_csv(os.path.join(directory, highest_version_file))

        # Concatenate the new data with the existing data
        df_combined = pd.concat([df, df2], ignore_index=True)

        # Save the combined data as a new version
        new_version = highest_version + 1
        new_version_file = f"depi_v{new_version}.csv"
        df_combined.to_csv(os.path.join(directory, new_version_file), index=False)

        # Process the combined DataFrame



        df_demand = preprocess_ungrouped(df_combined)
        df_series = grouping_data(df_demand)
        df_series = feature_engineering(df_series)
        df_series = time_series_features(df_series)

        df_demand = handle_category(df_demand)

        # df_combined = handle_unneded_columns(df_combined)
        # df_combined = handle_missing_columns(df_combined)
        # df_combined = handle_missing_rows(df_combined)
        # df_combined = feature_engineering(df_combined)


        df_old_marketing = pd.read_csv("data/marketing_data.csv")
        df_marketing_combined = pd.concat([df_old_marketing, df_marketing], ignore_index=True)

        # Save the updated DataFrame
        # df_combined.to_csv('Updated_Data.csv', index=False)

        st.subheader("Updated Data Preview")
        st.dataframe(df_combined.head())
        
        if st.button("Train Models"):
            # Train the models
            # train_sarimax_demand_forecasting(df_demand)
            print("Train Models button pressed")
            train_sarimax_sales_forecasting(df_series,df_marketing_combined)
            print("Passed the training")
            st.success("Models trained successfully!")
        # except FileNotFoundError:


        # except Exception as e:
        #     st.error(f"Error failed to upload file.")


# Main navigation logic
if page == "Forecasting Sales":
    forecasting_sales_page()
elif page == "Demand Product":
    demand_product_page()
elif page == "Visualizations":
    dashboards_page()
elif page == "Retraining models":
    re_training_models()


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Sales Forcasting & ON-Demand Predictions")
