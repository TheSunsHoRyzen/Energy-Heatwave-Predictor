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
    df['predicted_kWh'] = predict(df, artifacts)    
    return df


# === Process and Display ===
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()

        required_cols = {
            'date': 'date',
            'max temperature': 'TX',
            'min temperature': 'TN',
            'average temperature': 'TG',
            'sunshine duration': 'SS',
            'humidity': 'HU'
        }

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            # Rename columns
            df = df.rename(columns=required_cols)

            # Parse 'date' column
            try:
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
            except Exception as e:
                st.error(f"Error parsing dates: {e}")

            # Extract date features
            df['Year'] = df['date'].dt.year
            df['Month'] = df['date'].dt.month
            df['Day'] = df['date'].dt.day
            df['Weekday'] = df['date'].dt.weekday   # 1=Mon, ..., 7=Sun
            heatwave_threshold = df['TX'].quantile(0.95)
            df['Heatwave'] = (df['TX'] > heatwave_threshold).astype(int)

            numeric_cols = ['TX', 'TN', 'TG', 'HU', 'SS']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[~df.isna().any(axis=1)]  # drop rows with any missing values
            print(df.dtypes)


            # Predict
            print("Model expects features:", artifacts["features"])
            input_features = artifacts["features"]


            df['predicted_kWh'] = predict(df[input_features], artifacts)


            # df['temperature'] = df['TX']  # used in identify_heatwaves
            # Heatwave detection
            df = identify_heatwaves(df, threshold_temp=df['TX'].quantile(0.5), window=3, temp_col='TX')

            # === Display Table with Highlights ===
            st.markdown("### 🔍 Predicted Electricity Consumption (in kWh)")

            def highlight_heatwaves(row):
                group = row['heatwave_group']
                if group == 0:
                    return [''] * len(row)
                color = f'hsla({(group * 37) % 360}, 70%, 85%, 1)'
                return [f'background-color: {color}'] * len(row)

            styled_table = df.style\
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

            # === Heatwave Summary ===
            heatwave_groups = df[df['heatwave_group'] > 0]['heatwave_group'].unique()
            st.markdown(f"### 🌡️ Heatwave Summary: {len(heatwave_groups)} identified")

            if len(heatwave_groups) > 0:
                summary_rows = []
                for group_id in heatwave_groups:
                    sub_df = df[df['heatwave_group'] == group_id]
                    start_date = sub_df['date'].min().strftime('%m/%d/%Y')
                    end_date = sub_df['date'].max().strftime('%m/%d/%Y')
                    duration = (sub_df['date'].max() - sub_df['date'].min()).days + 1
                    mean_temp = sub_df['TX'].mean()
                    median_temp = sub_df['TX'].median()

                    summary_rows.append({
                        "Heatwave ID": group_id,
                        "Start Date": start_date,
                        "End Date": end_date,
                        "Duration (days)": duration,
                        "Mean Temp (°C)": round(mean_temp, 2),
                        "Median Temp (°C)": round(median_temp, 2)
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
