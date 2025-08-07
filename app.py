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

    # Identify heatwaves
    df = identify_heatwaves(df, threshold_temp=25, window=3, temp_col='temperature')
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
                st.markdown("### 🔍 Predicted Electricity Consumption (in kWh)")
                def highlight_heatwaves(row):
                    group = row['heatwave_group']
                    if group == 0:
                        return [''] * len(row)
                    # generate pastel color from group number
                    color = f'hsla({(group * 37) % 360}, 70%, 35%, 1)'
                    return [f'background-color: {color}'] * len(row)

                styled_table = result_df.style\
                    .format({
                        'date': lambda d: d.strftime('%m/%d/%Y'),
                        'temperature': '{:.1f} °C',
                        'predicted_kWh': '{:.2f} kWh',
                    })\
                    .apply(highlight_heatwaves, axis=1)\
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('background-color', '#003366'), ('color', 'white'), ('font-size', '14px')]
                    }])

            st.dataframe(styled_table, use_container_width=True)
            heatwave_groups = result_df[result_df['heatwave_group'] > 0]['heatwave_group'].unique()
            st.markdown(f"### 🌡️ Heatwave Summary: {len(heatwave_groups)} identified")

            if len(heatwave_groups) > 0:
                summary_rows = []
                for group_id in heatwave_groups:
                    sub_df = result_df[result_df['heatwave_group'] == group_id]
                    start_date = sub_df['date'].min().strftime('%m/%d/%Y')
                    end_date = sub_df['date'].max().strftime('%m/%d/%Y')
                    duration = (sub_df['date'].max() - sub_df['date'].min()).days + 1
                    mean_temp = sub_df['temperature'].mean()
                    median_temp = sub_df['temperature'].median()
                    low_temp = sub_df['temperature'].min()

                    summary_rows.append({
                        "Heatwave ID": group_id,
                        "Start Date": start_date,
                        "End Date": end_date,
                        "Duration (days)": duration,
                        "Mean Temp (°C)": round(mean_temp, 2),
                        "Median Temp (°C)": round(median_temp, 2),
                        "Minimum Temp (°C)": round(low_temp,2)
                    })

                st.dataframe(pd.DataFrame(summary_rows))


    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.markdown(
        "<p style='text-align: center; font-size: 1.1em; color: gray;'>"
        "Awaiting your data upload...</p>",
        unsafe_allow_html=True
    )
