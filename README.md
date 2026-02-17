# DemandOps | Intelligent Sales & Demand Forecasting 🚀

![Version](https://img.shields.io/badge/version-3.0-blue) ![Python](https://img.shields.io/badge/python-3.8+-green) ![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red) ![XGBoost](https://img.shields.io/badge/xgboost-1.7+-orange)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sales-and-demand-forecasting.streamlit.app/)

**DemandOps** is a professional-grade AI forecasting and demand planning platform designed for businesses to predict future sales, analyze trends, and simulate market scenarios. Built with **XGBoost** for precision modeling and **Streamlit** for an interactive dashboard, it empowers data-driven decision-making.

> 🌟 **[Live Demo: Click Here to Open App](https://sales-and-demand-forecasting.streamlit.app/)**

---

## 🌟 Key Features

*   **📈 Advanced Forecasting**: Utilizes XGBoost regression models to predict sales at daily, weekly, monthly, and quarterly granularities.
*   **🧠 Intelligent Feature Engineering**: Automatically extracts seasonal features, lag variables, and rolling averages to capture complex demand patterns.
*   **📊 Interactive Dashboard**: A sleek, dark-mode specialized UI built with Streamlit to visualize forecasts and historical data.
*   **🧪 Scenario Planning**: Simulate the impact of **Marketing Spend Uplifts** and **Market Conditions** (Boom/Recession) on your future revenue.
*   **📅 Seasonality Analysis**: Deep dive into daily and monthly buying habits to identify peak trading periods.
*   **📑 Automated Reporting**: Generates professional PDF executive summaries with embedded charts and insights.
*   **💾 Data Export**: Download forecast data in CSV format for offline analysis.

---

## 📂 Project Structure

```bash
📦 Sales-Demand-Forecasting
├── 📜 app.py               # Main Streamlit Dashboard Application
├── 📜 main.py              # Data Pipeline: Loading, Training, Forecasting, & Reporting
├── 📜 requirements.txt     # Python Dependencies
├── 📜 data.csv             # Source Data (Not included in repo, see Setup)
├── 📂 outputs/             # Generated Artefacts (CSVs, PNGs, PDF Reports)
├── 📜 README.md            # Project Documentation
└── 📜 .gitignore           # Git Configuration
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pranay9981/FUTURE_ML_01.git
cd FUTURE_ML_01
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
Place your sales data file named **`data.csv`** in the root directory.
*   **Required Columns**: `InvoiceNo`, `InvoiceDate`, `Quantity`, `UnitPrice`, `Country` (Standard Retail Dataset format).

---

## 💡 Usage Workflow

### Step 1: Run the Training Pipeline
Execute the main script to process data, train the XGBoost models, generate forecasts, and create the PDF report.

```bash
python main.py
```
*   **Output**: This will populate the `outputs/` directory with forecast CSVs (`forecast_daily.csv`, etc.), visualization images, and `sales_forecast_report.pdf`.
*   **Metrics**: Check the terminal output for MAE (Mean Absolute Error) and RMSLE scores.

### Step 2: Launch the Dashboard
Start the Streamlit application to explore the data interactively.

```bash
streamlit run app.py
```
*   The dashboard will open in your default browser (usually at `http://localhost:8501`).
*   **Explore**: Use the sidebar to toggle confidence intervals, adjust marketing spend, and switch between "Stable", "Booming", or "Recession" market scenarios.

---

## 📊 Dashboard Modules

1.  **Executive Summary**: High-level KPIs, Next Month Forecast, and Risk Assessment.
2.  **Forecast Deep Dive**: Granular time-series analysis (Daily/Weekly/Monthly) with zoomable charts.
3.  **Seasonality Patterns**: Visualizations of weekly buying habits and annual seasonality trends.
4.  **Data Export**: tailored CSV downloads for your external reporting needs.

---

## 🛠 Technologies Used

*   **Core**: Python, Pandas, NumPy
*   **Modeling**: XGBoost, Scikit-Learn
*   **Visualization**: Plotly, Matplotlib
*   **App Framework**: Streamlit
*   **Reporting**: ReportLab (PDF Generation)

---

## 👨‍💻 Author

**Built by Pranay Bagaria**
*   [GitHub](https://github.com/pranay9981)
*   [LinkedIn](https://www.linkedin.com/in/pranay-bagaria/)
