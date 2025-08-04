# model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_model():
    # --- Data load & merge ---
    weather = pd.read_csv("london_weather_data_1979_to_2023.csv")
    energy = pd.read_csv("london_energy.csv")
    
    # convert to °C
    for col in ('TG','TN','TX'):
        weather[col] = weather[col] / 10
    
    weather['date'] = pd.to_datetime(weather['DATE'], format='%Y%m%d')
    
    avg_kwh = energy.groupby('Date')['KWH'].mean()
    avg_kwh = pd.DataFrame({
        'date': avg_kwh.index.tolist(),
        'consumption': avg_kwh.values.tolist()
    })
    avg_kwh['date'] = pd.to_datetime(avg_kwh['date'])
    
    df = pd.merge(avg_kwh, weather.drop(columns=['DATE']), on='date', how='inner')
    
    # forward-fill any missing HU/CC
    df[['HU','CC']] = df[['HU','CC']].ffill()
    
    # --- Feature prep & PCA ---
    X = df[['TG','TN','TX']]  # Keep all three temperature features
    y = df['consumption']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)  # Back to 3 components
    X_pca = pca.fit_transform(X_scaled)
    
    # --- Train-test split & RF fit ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    
    rf = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=200,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    return {"model": rf, "scaler": scaler, "pca": pca}

def predict(feature_df: pd.DataFrame, artefacts: dict) -> np.ndarray:
    """
    feature_df: DataFrame with 'temperature' column (single temperature value)
    artefacts: dict from load_model()
    
    This function maps the single temperature input to TG, TN, TX format
    by using the temperature as average (TG) and estimating min/max
    """
    # Convert single temperature to three-feature format
    # Assume user temperature is average temp (TG)
    # Estimate TN (min) as TG - 5°C and TX (max) as TG + 5°C
    temp_values = feature_df['temperature'].values
    
    # Create three-column DataFrame matching training format
    expanded_features = pd.DataFrame({
        'TG': temp_values,           # Average temp (user input)
        'TN': temp_values - 5,       # Estimated min temp
        'TX': temp_values + 5        # Estimated max temp
    })
    
    X_scaled = artefacts["scaler"].transform(expanded_features)
    X_pca = artefacts["pca"].transform(X_scaled)
    return artefacts["model"].predict(X_pca)

def identify_heatwaves(df: pd.DataFrame, threshold_temp=25, window=3) -> pd.DataFrame:
    df = df.copy()
    
    # Use the temperature column instead of TX
    is_hot = df['temperature'] > threshold_temp
    streak_counts = np.convolve(is_hot, np.ones(window, dtype=int), mode='same')
    df["is_heatwave"] = streak_counts >= window
    
    starts = df["is_heatwave"] & ~df["is_heatwave"].shift(fill_value=False)
    df["heatwave_group"] = starts.cumsum() * df["is_heatwave"]
    
    return df