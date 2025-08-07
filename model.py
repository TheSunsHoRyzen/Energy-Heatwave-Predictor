# model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    # --- Load & merge your weather + energy data as before ---
    weather = pd.read_csv("london_weather_data_1979_to_2023.csv")
    energy  = pd.read_csv("london_energy.csv")

    # parse dates (specify format so no warning):
    weather['date'] = pd.to_datetime(weather['DATE'], format="%Y%m%d", errors="coerce")
    energy_dates   = pd.to_datetime(energy['Date'], format="%Y-%m-%d", errors="coerce")
    numerical_cols = weather.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        weather[col] = weather[col].fillna(weather[col].median())

    avg_kwh = (energy
              .assign(Date=energy_dates)
              .groupby('Date')['KWH']
              .mean()
              .reset_index(name='consumption')
              .rename(columns={'Date':'date'}))

    df = weather.drop(columns=['DATE']).merge(avg_kwh, on='date', how='inner')

    # fill missing HU/CC
    # df[['HU','CC']] = df[['HU','CC']].ffill()


    # convert temps from tenths of °C
    df[['TX','TN','TG']] = df[['TX','TN','TG']] / 10

    heatwave_threshold = df['TX'].quantile(0.95)
    df['Heatwave'] = (df['TX'] > heatwave_threshold).astype(int)
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Weekday'] = df['date'].dt.weekday
    # train on just the one TX feature
    features = ['TX', 'TN', 'TG', 'HU', 'SS', 'Heatwave', 'Year', 'Month', 'Day','Weekday']
            # Check for any object or string columns — there should be none

    X = df[features]
    y = df['consumption']

    # scaler = StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    # model = LinearRegression()
    # model.fit(X_tr,y_tr)
    # return {"model": rf, "scaler": scaler}
    return {"model": rf, "features": features}

    # return {"model": model}


def predict(feature_df: pd.DataFrame, artefacts: dict) -> np.ndarray:
    """
    feature_df: DataFrame with a 'temperature' column
    artefacts:   dict returned by load_model()
    """
    # work on a copy, rename 1→TX without warning]

    # heatwave_predictions = identify_heatwaves(df_local)

    # scale exactly like training
    # Xs = artefacts['scaler'].transform(df_local)

    # predict
    # predictions = artefacts['model'].predict(feature_df)
    


    features = artefacts['features']
    X = feature_df[features]
    return artefacts['model'].predict(feature_df)


    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=df_local['TX'], y=predictions, color='r')
    # plt.title('Linear Regression Predictions vs. TX')
    # plt.xlabel('TX (degrees C)')
    # plt.ylabel('Predicted Consumption (KWH)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("predictions_vs_temperature.png")  # Optional: save to file
    return predictions

def identify_heatwaves(
    df: pd.DataFrame,
    threshold_temp: float = 25,
    window: int = 3,
    temp_col: str = "TX"
) -> pd.DataFrame:
    df = df.copy()

    # Identify hot days
    df["is_hot"] = df[temp_col] > threshold_temp

    # Create group numbers for consecutive days (hot or not)
    group = (df["is_hot"] != df["is_hot"].shift()).cumsum()

    # Assign group ids only to hot streaks
    df["group"] = group.where(df["is_hot"])

    # Filter out groups shorter than the required window
    heatwave_counts = df.groupby("group").size()
    valid_groups = heatwave_counts[heatwave_counts >= window].index

    # Assign heatwave_group id (incremental starting from 1)
    df["heatwave_group"] = 0
    heatwave_id = 1
    for g in valid_groups:
        df.loc[df["group"] == g, "heatwave_group"] = heatwave_id
        heatwave_id += 1

    # Flag is_heatwave based on group assignment
    df["is_heatwave"] = df["heatwave_group"] > 0

    # Clean up
    df.drop(columns=["group", "is_hot"], inplace=True)
    return df

