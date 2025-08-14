# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from model import load_model, predict, identify_heatwaves
import plotly.graph_objects as go

# === Page Config ===
st.set_page_config(page_title="Electricity Consumption Predictor for London", layout="centered")

# === Title ===
st.markdown("""
    <div style="
        background: linear-gradient(360deg, #48288c 0%, #db7f4b 90%);
        padding: 2rem 1rem 1.2rem 1rem;
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style='color: #fff; font-size: 2.7em; font-family:Segoe UI,Arial,sans-serif; margin-bottom: 0.2em; letter-spacing: 1px;'>
             Electricity Consumption Predictor for London
        </h1>
        <p style='color: #e0e0e0; font-size: 1.25em; margin-top: 0;'>
            <em>Upload your temperature, sunshine, and humidity data to forecast daily electricity usage (in kWh)<em>
        </p>
    </div>
""", unsafe_allow_html=True)

# === File Upload ===
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    label_visibility="collapsed",  
    help="File should contain 'date' (MM/DD/YYYY) and temperature columns"
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
            heatwave_threshold = df['TX'].quantile(0.80)
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
            df = identify_heatwaves(df, threshold_temp=df['TX'].quantile(0.80), window=3, temp_col='TX')

            # === Display Table with Highlights ===
            st.markdown("### 🔍 Predicted Electricity Consumption (in kWh)")

            def highlight_heatwaves(row):
                group = row['heatwave_group']
                if group == 0:
                    return [''] * len(row)
                return [f'background-color: #db7f4b; color: white'] * len(row)
            

            styled_table = df.style\
                .format({
                    'date': lambda d: d.strftime('%m/%d/%Y'),
                    'TX': '{:.1f}',           # Max temperature
                    'TN': '{:.1f}',           # Min temperature
                    'TG': '{:.1f}',           # Avg temperature
                    'HU': '{:.0f}',           # Humidity as integer
                    'SS': '{:.1f}',           # Sunshine duration
                    'predicted_kWh': '{:.2f}',# Predicted kWh
                
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
                    min_temp = sub_df['TX'].min()
                    max_temp = sub_df['TX'].max()
                    total_pred_kwh = sub_df['predicted_kWh'].sum()
                    mean_pred_kwh = sub_df['predicted_kWh'].mean()
                    median_pred_kwh = sub_df['predicted_kWh'].median()
                    start_weekday = sub_df['date'].min().strftime('%A')
                    end_weekday = sub_df['date'].max().strftime('%A')
                    dates_str = ', '.join(sub_df['date'].dt.strftime('%m/%d/%Y'))

                    summary_rows.append({
                        "Heatwave ID": group_id,
                        "Start Date": start_date,
                        "End Date": end_date,
                        "Start Weekday": start_weekday,
                        "End Weekday": end_weekday,
                        "Duration (days)": duration,
                        "Min Temp Of Max Temperatures(°C)": round(min_temp, 2),
                        "Mean Temp Of Max Temperatures (°C)": round(mean_temp, 2),
                        "Median Temp Of Max Temperatures (°C)": round(median_temp, 2),
                        "Max Temp of Max Temperatures (°C)": round(max_temp, 2),
                        "Total Pred. kWh": round(total_pred_kwh, 2),
                        "Mean Pred. kWh": round(mean_pred_kwh, 2),
                        "Median Pred. kWh": round(median_pred_kwh, 2),
                        "Dates": dates_str,  
                    })

                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df)

                # === Heatwave Statistics ===
                avg_duration = summary_df["Duration (days)"].mean()
                max_duration = summary_df["Duration (days)"].max()
                min_duration = summary_df["Duration (days)"].min()
                highest_temp = summary_df["Max Temp of Max Temperatures (°C)"].max()
                total_heatwave_days = summary_df["Duration (days)"].sum()
                total_heatwaves = len(summary_df)
                total_pred_kwh = summary_df["Total Pred. kWh"].sum()

                # === Display Metrics ===
                st.markdown("### 📊 Heatwave Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Heatwave Events", total_heatwaves)
                col2.metric("Total Heatwave Days", int(total_heatwave_days))
                col3.metric("Max Temp (°C)", f"{highest_temp:.1f}")
                col4.metric("Total kWh (Heatwaves)", f"{total_pred_kwh:.2f}")

                st.markdown("### Heatwave Statistics")
                st.markdown(f"""
                - **Total heatwave events:** {total_heatwaves}
                - **Total heatwave days:** {int(total_heatwave_days)}
                - **Average heatwave duration:** {avg_duration:.1f} days
                - **Longest heatwave:** {int(max_duration)} days
                - **Shortest heatwave:** {int(min_duration)} days
                - **Highest temperature during heatwaves:** {highest_temp:.1f}°C
                - **Total predicted consumption during heatwaves:** {total_pred_kwh:.2f} kWh
                """)
                
                # === Plots ===
                st.markdown("### 📈 Visualizations") 
                  
                fig = px.bar(
                summary_df,
                x='Heatwave ID',
                y='Total Pred. kWh',
                color='Heatwave ID',  
                hover_data=['Duration (days)', 'Max Temp of Max Temperatures (°C)'],
                title="Total Predicted Consumption per Heatwave Event")
                st.plotly_chart(fig, use_container_width=True)
                
                # fig2 = px.line(
                # df, x='date', y='predicted_kWh', color='heatwave_group',
                # title="Predicted Electricity Consumption Over Time",
                # labels={'predicted_kWh': 'Predicted kWh', 'date': 'Date', 'heatwave_group': 'Heatwave'})
                # st.plotly_chart(fig2, use_container_width=True)

                # Sort by date to ensure continuity
                df_sorted = df.sort_values(by='date').reset_index(drop=True)

                # Find change points in heatwave group
                change_points = (df_sorted['heatwave_group'] != df_sorted['heatwave_group'].shift()).cumsum()

                # Create a figure
                fig = go.Figure()

                # Map group id to color
                color_map = {
                    0: 'steelblue',  # Normal
                    1: '#db7f4b',    # Heatwave 1
                    2: '#c44536',
                    3: '#e49b0f',
                    4: '#d7263d',
                    5: '#a30000',
                    6: '#f46036'
                }
                default_color = '#808080'  # fallback color

                # Plot continuous segments with same heatwave group
                for _, segment in df_sorted.groupby(change_points):
                    group = segment['heatwave_group'].iloc[0]
                    color = color_map.get(group, default_color)

                    fig.add_trace(go.Scatter(
                        x=segment['date'],
                        y=segment['predicted_kWh'],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f"Heatwave {group}" if group > 0 else "Normal",
                        showlegend=False  # Avoid legend spam; you can set this to True for the first occurrence only
                    ))

                # Final layout
                fig.update_layout(
                    title="Predicted Electricity Consumption Over Time",
                    xaxis_title="Date",
                    yaxis_title="Predicted kWh",
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)


                fig3 = px.scatter(
                df,
                x="TG",  
                y="predicted_kWh",
                color="heatwave_group",  
                labels={"TG": "Average Temp (°C)", "predicted_kWh": "Predicted kWh", "heatwave_group": "Heatwave"},
                title="Average Temperature (TG) vs Predicted Electricity Consumption"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # === Download Summary ===
            st.markdown("### 📥 Download Heatwave Summary")
            st.download_button("Download Table as CSV", summary_df.to_csv(index=False), "heatwave_summary.csv")
                


    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")



