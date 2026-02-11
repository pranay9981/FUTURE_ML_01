import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from datetime import date
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import warnings

warnings.filterwarnings("ignore")

# ===================================================
# OUTPUT DIRECTORY
# ===================================================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================
# 1. LOAD & CLEAN
# ===================================================
def load_and_process_data(filepath):
    print("Loading data...")
    # Try different encodings
    try:
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
    except:
        df = pd.read_csv(filepath, encoding="utf-8")
        
    df.columns = df.columns.str.strip()
    if 'InvoiceNo' not in df.columns:
        df.rename(columns={df.columns[0]: 'InvoiceNo'}, inplace=True)

    # Filter for UK and valid sales
    df = df[
        (df['Country'] == 'United Kingdom') &
        (df['Quantity'] > 0) &
        (df['UnitPrice'] > 0)
    ]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
    df['Date'] = df['InvoiceDate'].dt.date
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    # Aggregating to Daily
    daily = df.groupby('Date').agg(
        TotalSales=('Sales', 'sum'),
        InvoiceCount=('InvoiceNo', 'nunique'),
        ItemCount=('Quantity', 'sum')
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq('D')
    
    # Fill missing days with 0
    daily[['TotalSales', 'InvoiceCount', 'ItemCount']] = daily[['TotalSales', 'InvoiceCount', 'ItemCount']].fillna(0)

    # Calculate AOV
    daily['AOV'] = np.where(
        daily['InvoiceCount'] > 0,
        daily['TotalSales'] / daily['InvoiceCount'],
        0
    )
    
    return daily

# ===================================================
# 2. FEATURES
# ===================================================
def days_to_xmas(d):
    x = date(d.year, 12, 25)
    if d.date() > x:
        x = date(d.year + 1, 12, 25)
    return (x - d.date()).days

def create_features(df):
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['days_to_xmas'] = df.index.map(days_to_xmas)
    
    # Lag features
    df['lag_items_1'] = df['ItemCount'].shift(1)
    df['lag_items_7'] = df['ItemCount'].shift(7)
    df['roll_items_7'] = df['ItemCount'].shift(1).rolling(7).mean()
    
    df['lag_aov_1'] = df['AOV'].shift(1)
    df['lag_aov_7'] = df['AOV'].shift(7)
    df['roll_aov_7'] = df['AOV'].shift(1).rolling(7).mean()
    
    return df.dropna()

# ===================================================
# 3. TRAIN
# ===================================================
def train_models(df, test_days=60):
    print("Training models...")
    split_date = df.index.max() - pd.Timedelta(days=test_days)
    train = df[df.index <= split_date]
    test = df[df.index > split_date]

    # --- Item Count Model ---
    feats_items = ['day_of_week', 'month', 'is_weekend', 'days_to_xmas', 'lag_items_1', 'lag_items_7', 'roll_items_7']
    model_items = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, objective='reg:squarederror')
    model_items.fit(train[feats_items], np.log1p(train['ItemCount']))
    
    # Predict Items
    test_items = np.expm1(model_items.predict(test[feats_items]))
    train_items = np.expm1(model_items.predict(train[feats_items]))

    # --- AOV Model ---
    feats_aov = ['day_of_week', 'month', 'is_weekend', 'days_to_xmas', 'lag_aov_1', 'lag_aov_7', 'roll_aov_7']
    model_aov = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, objective='reg:squarederror')
    model_aov.fit(train[feats_aov], train['AOV'])
    
    # Predict AOV
    test_aov = np.maximum(0, model_aov.predict(test[feats_aov]))
    train_aov = np.maximum(0, model_aov.predict(train[feats_aov]))

    # --- Combine for Sales ---
    # We use a stacked approach: base_sales = items * aov
    train = train.copy()
    test = test.copy()
    train['base_sales'] = train_items * train_aov
    test['base_sales'] = test_items * test_aov
    
    sales_feats = ['base_sales', 'day_of_week', 'month', 'is_weekend']
    model_sales = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, objective='reg:squarederror')
    model_sales.fit(train[sales_feats], train['TotalSales'])
    
    # Final Predictions
    preds = np.maximum(0, model_sales.predict(test[sales_feats]))
    
    return train, test, preds

# ===================================================
# 4. SCENARIO GENERATION & SAVING
# ===================================================
def save_forecast_data(full_df, freq, name):
    """Saves the forecast data to CSV for the dashboard to read."""
    df = full_df.copy()
    
    # Calculate residuals for uncertainty bounds
    residuals = df['TotalSales'] - df['Forecast']
    sigma = residuals.dropna().std()
    
    df['Best'] = df['Forecast'] + 1.96 * sigma
    df['Worst'] = np.maximum(0, df['Forecast'] - 1.96 * sigma)
    
    # Resample if needed
    if freq:
        df = df.resample(freq).sum()
    
    # Save to CSV
    filename = f"{OUTPUT_DIR}/forecast_{name.lower()}.csv"
    df.to_csv(filename)
    print(f"✔ Saved data: {filename}")

# ===================================================
# MAIN EXECUTION
# ===================================================
if __name__ == "__main__":
    # Update this path to where your file is located
    file_path = "D:\\Sales & Demand Forecasting for Businesses\\data.csv"
    
    if os.path.exists(file_path):
        data = load_and_process_data(file_path)
        features = create_features(data)
        
        # Train
        train, test, preds = train_models(features)
        
        # Calculate Metrics
        mae = mean_absolute_error(test['TotalSales'], preds)
        print(f"Model MAE: £{mae:.2f}")

        # Create Full Forecast Timeline
        forecast_series = pd.concat([
            data[['TotalSales']], 
            test[['TotalSales']].assign(TotalSales=preds)
        ]).sort_index()
        
        forecast_series.rename(columns={'TotalSales': 'Forecast'}, inplace=True)
        
        # Join Actuals with Forecast
        full_df = data.join(forecast_series, how='left')
        
        # In the 'test' period, the 'TotalSales' column is the Actual, 'Forecast' is Predicted.
        # We need to ensure the forecast column covers the future.
        # For this demo, we are "forecasting" the test set.
        full_df['Forecast'] = full_df['Forecast'].fillna(full_df['TotalSales'])

        # --- SAVE DATA FOR DASHBOARD ---
        save_forecast_data(full_df, None, "Daily")
        save_forecast_data(full_df, "W", "Weekly")
        save_forecast_data(full_df, "M", "Monthly")
        save_forecast_data(full_df, "Q", "Quarterly")
        
        print("\nSUCCESS: Forecast data generated. Now run dashboard.py")
        
    else:
        print(f"Error: File not found at {file_path}")