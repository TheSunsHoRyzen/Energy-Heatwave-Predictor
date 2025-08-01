import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



# def identify_heatwaves():



df = pd.read_csv("london_weather_data_1979_to_2023.csv")
df2 = pd.read_csv("london_energy.csv")

# Convert temperature from 0.1°C to °C
df['TX'] = df['TX'] / 10
df['TN'] = df['TN'] / 10
df['TG'] = df['TG'] / 10


avg_kwh = df2.groupby('Date')['KWH'].mean()
avg_kwh = pd.DataFrame({'date':avg_kwh.index.tolist(), 'consumption':avg_kwh.values.tolist()})
avg_kwh['date'] = pd.to_datetime(avg_kwh['date'])

# Step 2: Convert the weather data 'DATE' column to datetime
# Keep DATE column as datetime (do not convert to string)
df['date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')


merged_df = pd.merge(avg_kwh, df.drop(columns=['DATE']), on='date', how='inner')

print(merged_df.isna().any()[lambda x: x])
print(merged_df.info())
print(merged_df.head())
print(f"Number of null values: {merged_df.isnull().sum().sum()}")

columns_to_fill = ['HU', 'CC']
merged_df[columns_to_fill] = merged_df[columns_to_fill].ffill()

print(merged_df.info())
df = merged_df


# PCA for dimensionality reduction
X = df[['TG', 'TN', 'TX']]
y = df['consumption']

# Standardize features pre-PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA (all 3 comps)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Split PCA features
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, train_size=0.8, random_state=42)

# Linear Regression w/ PCA features
# model_pca = LinearRegression()
# model_pca.fit(X_train_pca, y_train)
# y_pred_linear_pca = model_pca.predict(X_test_pca)

# print("Linear Regression with PCA features")
# print("MSE:", mean_squared_error(y_test, y_pred_linear_pca))
# print("R²:", r2_score(y_test, y_pred_linear_pca))

# Random Forest w/ PCA features
rf_pca = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_rf_pca = rf_pca.predict(X_test_pca)

print("Random Forest Regression with PCA features")
print("MSE:", mean_squared_error(y_test, y_pred_rf_pca))
print("R²:", r2_score(y_test, y_pred_rf_pca))
