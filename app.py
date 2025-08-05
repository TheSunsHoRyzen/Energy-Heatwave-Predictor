# app.py
import streamlit as st
import pandas as pd
from model import load_model, predict, identify_heatwaves

# === Page Config ===
st.set_page_config(page_title="Electricity Consumption Predictor for London", layout="centered")

# === Title ===
st.markdown("""
    <h1 style='text-align: center; color: #004080; font-size: 3em;'>
        ⚡ Electricity Consumption Predictor for London
    </h1>
    <p style='text-align: center; color: #666; font-size: 1.2em;'>
        Upload your temperature data to forecast daily electricity usage (in kWh)
    </p>
""", unsafe_allow_html=True)

# === File Upload ===
uploaded_file = st.file_uploader(
    "Upload CSV with 'date' and 'temperature' columns",
    type=['csv'],
    help="File should contain 'date' (MM/DD/YYYY) and 'temperature' columns"
)

# === Load & Cache Model Once ===
@st.cache_resource
def get_model_artifacts():
    return load_model()

artifacts = get_model_artifacts()

# === Prediction Helper ===
def predict_consumption(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['predicted_kWh'] = predict(df[['temperature']], artifacts)
    return df

# === Process and Display ===
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()

        if 'temperature' in df.columns:
            temp_col = 'temperature'
        elif 'tx' in df.columns:
            temp_col = 'tx'
            df.rename(columns={'tx': 'temperature'}, inplace=True)
        else:
            temp_col = None

        # check for 'date' column and parse format
        if 'date' in df.columns:
            # parse YYYYMMDD or MM/DD/YYYY
            if df['date'].astype(str).str.match(r'^\d{8}$').all():
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            else:
                # fallback: try MM/DD/YYYY
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        else:
            st.error("CSV must contain a 'date' column.")
            temp_col = None

        # validate columns
        if temp_col is None or 'date' not in df.columns:
            st.error("CSV must contain 'date' (YYYYMMDD or MM/DD/YYYY) and 'temperature' or 'tx' columns.")
        else:
            df = df.dropna(subset=['date', 'temperature'])
            if df.empty:
                st.error(
                    "No valid rows to predict. "
                    "Please check your 'date' and 'temperature' columns for correct format and missing values."
                )
            else:
                input_df  = df[['date', 'temperature']]
                result_df = predict_consumption(input_df)

                # === Display Result Table ===
                st.markdown("### 🔍 Predicted Electricity Consumption (in kWh)")
                styled_table = result_df.style.format({
                    'date': lambda d: d.strftime('%m/%d/%Y'),
                    'temperature': '{:.1f} °C',
                    'predicted_kWh': '{:.2f} kWh'
                }).set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#003366'),
                        ('color', 'white'),
                        ('font-size', '14px')
                    ]}
                ]).highlight_max(color='lightgreen')

                st.dataframe(styled_table, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.markdown(
        "<p style='text-align: center; font-size: 1.1em; color: gray;'>"
        "Awaiting your data upload...</p>",
        unsafe_allow_html=True
    )
