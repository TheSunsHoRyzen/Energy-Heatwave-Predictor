import streamlit as st
import pandas as pd
from datetime import datetime
from model import predict

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
uploaded_file = st.file_uploader("Upload CSV with 'date' and 'temperature' columns", type=['csv'],
                                 help="File should contain 'date' and 'temperature' columns")

# === Dummy Model Function ===
def predict_consumption(df):
    # Replace with your actual model logic
    # Example: electricity = a * temperature + b
    df['predicted_kWh'] = predict(df['temperature'])

    return df

# === Process and Display ===
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        if 'date' not in df.columns or 'temperature' not in df.columns:
            st.error("CSV must contain 'date' and 'temperature' columns.")
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'temperature'], inplace=True)

            result_df = predict_consumption(df[['date', 'temperature']])

            # === Display Result Table ===
            st.markdown("### 🔍 Predicted Electricity Consumption (in kWh)")
            styled_table = result_df.style.format({
                'date': lambda d: d.strftime('%Y-%m-%d'),
                'temperature': '{:.1f} °C',
                'predicted_kWh': '{:.2f} kWh'
            }).set_table_styles(
                [{'selector': 'th', 'props': [('background-color', '#003366'), ('color', 'white'), ('font-size', '14px')]}]
            ).highlight_max(color='lightgreen')


            st.dataframe(styled_table, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: gray;'>Awaiting your data upload...</p>", unsafe_allow_html=True)
