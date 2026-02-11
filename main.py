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
# STEP 1: LOAD & CLEAN DATA
# ===================================================
def load_and_process_data(filepath):
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()

    if 'InvoiceNo' not in df.columns:
        df.rename(columns={df.columns[0]: 'InvoiceNo'}, inplace=True)

    df = df[
        (df['Country'] == 'United Kingdom') &
        (df['Quantity'] > 0) &
        (df['UnitPrice'] > 0)
    ]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
    df['Date'] = df['InvoiceDate'].dt.date
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    daily = df.groupby('Date').agg(
        TotalSales=('Sales', 'sum'),
        InvoiceCount=('InvoiceNo', 'nunique'),
        ItemCount=('Quantity', 'sum')
    )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq('D')

    daily[['TotalSales', 'InvoiceCount', 'ItemCount']] = \
        daily[['TotalSales', 'InvoiceCount', 'ItemCount']].fillna(0)

    daily['AOV'] = np.where(
        daily['InvoiceCount'] > 0,
        daily['TotalSales'] / daily['InvoiceCount'],
        np.nan
    )
    daily['AOV'] = daily['AOV'].ffill()

    return daily

# ===================================================
# STEP 2: FEATURE ENGINEERING
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

    df['is_bulk_day'] = (df['ItemCount'] > df['ItemCount'].quantile(0.9)).astype(int)

    df['lag_items_1'] = df['ItemCount'].shift(1)
    df['lag_items_7'] = df['ItemCount'].shift(7)
    df['roll_items_7'] = df['ItemCount'].shift(1).rolling(7).mean()

    df['lag_aov_1'] = df['AOV'].shift(1)
    df['lag_aov_7'] = df['AOV'].shift(7)
    df['roll_aov_7'] = df['AOV'].shift(1).rolling(7).mean()

    return df.dropna()

# ===================================================
# STEP 3: TRAIN & FORECAST
# ===================================================
def train_models(df, test_days=30):
    split_date = df.index.max() - pd.Timedelta(days=test_days)
    train = df[df.index <= split_date]
    test = df[df.index > split_date]

    feats_items = [
        'day_of_week', 'month', 'is_weekend', 'days_to_xmas',
        'lag_items_1', 'lag_items_7', 'roll_items_7'
    ]
    feats_aov = [
        'day_of_week', 'month', 'is_weekend', 'days_to_xmas',
        'lag_aov_1', 'lag_aov_7', 'roll_aov_7'
    ]

    model_items = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror'
    )
    model_items.fit(train[feats_items], np.log1p(train['ItemCount']))

    model_aov = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        objective='reg:squarederror'
    )
    mask = train['InvoiceCount'] > 0
    model_aov.fit(train.loc[mask, feats_aov], train.loc[mask, 'AOV'])

    train_items = np.expm1(model_items.predict(train[feats_items]))
    test_items = np.expm1(model_items.predict(test[feats_items]))

    train_aov = np.clip(model_aov.predict(train[feats_aov]), 0, None)
    test_aov = np.clip(model_aov.predict(test[feats_aov]), 0, None)

    train = train.copy()
    test = test.copy()

    train['base_sales'] = train_items * train_aov
    test['base_sales'] = test_items * test_aov

    train['log_base_sales'] = np.log1p(train['base_sales'])
    test['log_base_sales'] = np.log1p(test['base_sales'])

    sales_feats = [
        'base_sales', 'log_base_sales',
        'day_of_week', 'month', 'is_weekend', 'is_bulk_day'
    ]

    model_sales = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        objective='reg:squarederror'
    )
    model_sales.fit(train[sales_feats], np.log1p(train['TotalSales']))

    preds = np.expm1(model_sales.predict(test[sales_feats]))

    return train, test, preds

# ===================================================
# STEP 4: SEASONALITY SCENARIO
# ===================================================
def seasonal_scenario(full_df):
    df = full_df.copy()
    monthly_avg = df.groupby(df.index.month)['TotalSales'].mean()
    overall_avg = df['TotalSales'].mean()
    seasonality_index = monthly_avg / overall_avg

    df['Seasonality'] = df.index.month.map(seasonality_index)
    df['Seasonal_Forecast'] = df['Forecast'] * df['Seasonality']

    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Forecast'], label="Base Case")
    plt.plot(df.index, df['Seasonal_Forecast'], '--', label="Seasonality Scenario")
    plt.title("Seasonality-Adjusted Forecast")
    plt.ylabel("Sales (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scenario_seasonality.png")
    plt.close()

# ===================================================
# STEP 5: BASE / BEST / WORST SCENARIOS
# ===================================================
def scenario_forecast(full_df, label, freq=None):
    df = full_df.copy()

    residuals = df['TotalSales'] - df['Forecast']
    sigma = residuals.dropna().std()

    df['Best'] = df['Forecast'] + 1.96 * sigma
    df['Worst'] = np.maximum(0, df['Forecast'] - 1.96 * sigma)

    if freq:
        df = df.resample(freq).sum()

    if df.empty:
        return

    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Forecast'], label="Base Case")
    plt.plot(df.index, df['Best'], '--', label="Best Case")
    plt.plot(df.index, df['Worst'], '--', label="Worst Case")
    plt.fill_between(df.index, df['Worst'], df['Best'], alpha=0.25)
    plt.title(f"{label} Scenario Forecast")
    plt.ylabel("Sales (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scenario_{label.lower()}.png")
    plt.close()

# ===================================================
# STEP 6: PDF REPORT (WITH GRAPH EXPLANATIONS)
# ===================================================
def generate_pdf_report(mae, rmsle):
    report_path = f"{OUTPUT_DIR}/sales_forecast_report.pdf"

    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # -------------------------------
    # TITLE
    # -------------------------------
    elements.append(Paragraph("Sales Forecasting & Scenario Analysis Report", styles['Title']))
    elements.append(Spacer(1, 14))

    # -------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------
    elements.append(Paragraph("<b>1. Executive Summary</b>", styles['Heading2']))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        f"""
        This report presents a comprehensive sales forecasting analysis using historical transaction data.
        The objective is to estimate future sales under multiple scenarios to support operational,
        financial, and strategic decision-making.
        <br/><br/>
        The forecasting system produces:
        <ul>
            <li>A <b>Base Case</b> representing the most likely sales outcome</li>
            <li>A <b>Best Case</b> reflecting strong demand conditions</li>
            <li>A <b>Worst Case</b> reflecting conservative demand assumptions</li>
            <li>A <b>Seasonality Scenario</b> that adjusts forecasts based on recurring seasonal patterns</li>
        </ul>
        <br/>
        <b>Model Accuracy:</b><br/>
        Mean Absolute Error (MAE): £{mae:,.2f}<br/>
        Root Mean Squared Log Error (RMSLE): {rmsle:.3f}
        <br/><br/>
        MAE represents the average daily forecast error in currency terms, while RMSLE
        measures proportional error and is particularly useful for evaluating performance
        during high-sales periods.
        """,
        styles['BodyText']
    ))

    elements.append(Spacer(1, 18))

    # -------------------------------
    # SCENARIO EXPLANATION
    # -------------------------------
    elements.append(Paragraph("<b>2. Forecast Scenarios Explained</b>", styles['Heading2']))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        """
        <b>Base Case:</b> The most likely forecast generated by the model using historical patterns.<br/><br/>
        <b>Best Case:</b> Represents an upside scenario where demand exceeds expectations. This is
        calculated by adding historical forecast volatility to the base forecast.<br/><br/>
        <b>Worst Case:</b> Represents a downside risk scenario where demand underperforms expectations.
        This is calculated by subtracting historical forecast volatility from the base forecast.<br/><br/>
        <b>Seasonality Scenario:</b> Adjusts the base forecast using historical seasonal patterns
        (e.g., holiday peaks or off-season slowdowns).
        """,
        styles['BodyText']
    ))

    elements.append(Spacer(1, 18))

    # -------------------------------
    # CHART SECTIONS
    # -------------------------------
    chart_explanations = [
        (
            "scenario_daily.png",
            "3. Daily Scenario Forecast",
            """
            <b>What this chart shows:</b><br/>
            Daily sales forecasts under Base, Best, and Worst scenarios.
            <br/><br/>
            <b>How to read it:</b><br/>
            The solid line represents the most likely (base) forecast.
            The dashed lines represent optimistic and conservative outcomes.
            The shaded area shows the uncertainty range.
            <br/><br/>
            <b>How to use it:</b><br/>
            Use this view for short-term operational planning such as staffing,
            promotions, and inventory replenishment.
            <br/><br/>
            <b>Important note:</b><br/>
            Daily values can be volatile; trends are more important than individual points.
            """
        ),
        (
            "scenario_weekly.png",
            "4. Weekly Scenario Forecast",
            """
            <b>What this chart shows:</b><br/>
            Weekly aggregated sales forecasts across scenarios.
            <br/><br/>
            <b>How to read it:</b><br/>
            Each point represents total sales for a full week, reducing daily noise.
            <br/><br/>
            <b>How to use it:</b><br/>
            Ideal for workforce planning, weekly targets, and short-term supply planning.
            """
        ),
        (
            "scenario_monthly.png",
            "5. Monthly Scenario Forecast",
            """
            <b>What this chart shows:</b><br/>
            Monthly sales forecasts aligned with financial reporting periods.
            <br/><br/>
            <b>How to read it:</b><br/>
            Values represent total sales for each calendar month.
            <br/><br/>
            <b>How to use it:</b><br/>
            Suitable for budget tracking, revenue forecasting, and management reporting.
            """
        ),
        (
            "scenario_quarterly.png",
            "6. Quarterly Scenario Forecast",
            """
            <b>What this chart shows:</b><br/>
            Quarterly sales projections under different demand scenarios.
            <br/><br/>
            <b>How to read it:</b><br/>
            Each point represents total sales for a full quarter.
            <br/><br/>
            <b>How to use it:</b><br/>
            Designed for strategic planning, board-level reporting, and long-term forecasting.
            """
        ),
        (
            "scenario_seasonality.png",
            "7. Seasonality-Adjusted Forecast",
            """
            <b>What this chart shows:</b><br/>
            A forecast adjusted for recurring seasonal patterns observed in historical data.
            <br/><br/>
            <b>How to read it:</b><br/>
            The seasonality-adjusted line highlights periods where sales are expected
            to systematically increase or decrease due to seasonality.
            <br/><br/>
            <b>How to use it:</b><br/>
            Useful for planning around holidays, peak seasons, and off-peak periods.
            """
        ),
    ]

    for img, title, explanation in chart_explanations:
        path = f"{OUTPUT_DIR}/{img}"
        if os.path.exists(path):
            elements.append(Spacer(1, 14))
            elements.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(explanation, styles['BodyText']))
            elements.append(Spacer(1, 10))
            elements.append(Image(path, width=460, height=260))

    doc.build(elements)
    print("✔ Fully documented executive PDF generated")

# ===================================================
# MAIN
# ===================================================
if __name__ == "__main__":
    data = load_and_process_data("data.csv")
    features = create_features(data)

    train, test, preds = train_models(features)

    mae = mean_absolute_error(test['TotalSales'], preds)
    rmsle = np.sqrt(mean_squared_log_error(test['TotalSales'] + 1, preds + 1))

    print("\n--- FINAL METRICS ---")
    print(f"MAE   : £{mae:,.2f}")
    print(f"RMSLE : {rmsle:.3f}")

    # Build full timeline
    forecast_series = pd.concat([
        data[['TotalSales']],
        test[['TotalSales']].assign(TotalSales=preds)
    ]).sort_index()

    forecast_series = forecast_series.rename(columns={'TotalSales': 'Forecast'})
    full_df = data.join(forecast_series, how='left')

    # Scenarios
    scenario_forecast(full_df, "Daily")
    scenario_forecast(full_df, "Weekly", freq="W")
    scenario_forecast(full_df, "Monthly", freq="ME")
    scenario_forecast(full_df, "Quarterly", freq="QE")
    seasonal_scenario(full_df)

    # Report
    generate_pdf_report(mae, rmsle)

    print("\n✔ All scenarios generated")
    print("✔ Detailed executive PDF report created")