# model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_model():
    # --- Load & merge your weather + energy data as before ---
    weather = pd.read_csv("london_weather_data_1979_to_2023.csv")
    energy  = pd.read_csv("london_energy.csv")

    # parse dates (specify format so no warning):
    weather['date'] = pd.to_datetime(weather['DATE'], format="%Y%m%d", errors="coerce")
    energy_dates   = pd.to_datetime(energy['Date'], format="%Y-%m-%d", errors="coerce")

    avg_kwh = (energy
              .assign(Date=energy_dates)
              .groupby('Date')['KWH']
              .mean()
              .reset_index(name='consumption')
              .rename(columns={'Date':'date'}))

    df = weather.drop(columns=['DATE']).merge(avg_kwh, on='date', how='inner')

    # fill missing HU/CC
    df[['HU','CC']] = df[['HU','CC']].ffill()

    # convert temps from tenths of °C
    df[['TX','TN','TG']] = df[['TX','TN','TG']] / 10

    # train on just the one TX feature
    X = df[['TX']]
    y = df['consumption']

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=200,
        random_state=42
    )
    rf.fit(X_tr, y_tr)

    return {"model": rf, "scaler": scaler}


def predict(feature_df: pd.DataFrame, artefacts: dict) -> np.ndarray:
    """
    feature_df: DataFrame with a 'temperature' column
    artefacts:   dict returned by load_model()
    """
    # work on a copy, rename 1→TX without warning
    df_local = feature_df.rename(columns={'temperature':'TX'})[['TX']]

    # scale exactly like training
    Xs = artefacts['scaler'].transform(df_local)

    # predict
    return artefacts['model'].predict(Xs)


def identify_heatwaves(
    df: pd.DataFrame,
    threshold_temp: float = 25,
    window: int = 3
) -> pd.DataFrame:
    df = df.copy()
    is_hot = df['temperature'] > threshold_temp
    streak = np.convolve(is_hot, np.ones(window, dtype=int), mode='same')
    df["is_heatwave"]    = streak >= window
    starts = df["is_heatwave"] & ~df["is_heatwave"].shift(fill_value=False)
    df["heatwave_group"] = starts.cumsum() * df["is_heatwave"]
    return df
