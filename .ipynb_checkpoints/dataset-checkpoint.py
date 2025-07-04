import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("london_weather_data_1979_to_2023.csv")
df2 = pd.read_csv("london_energy.csv")


# print(df.info())

avg_kwh = df2.groupby('Date')['KWH'].mean()
avg_kwh = pd.DataFrame({'date':avg_kwh.index.tolist(), 'consumption':avg_kwh.values.tolist()})
avg_kwh['date'] = pd.to_datetime(avg_kwh['date'])
# print(avg_kwh.head())
# Step 2: Convert the weather data 'DATE' column to datetime
# Keep DATE column as datetime (do not convert to string)
df['date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# Merge on 'date' column only
merged_df = pd.merge(avg_kwh, df.drop(columns=['DATE']), on='date', how='inner')


# Output the shape of the merged result for quick verification
# print(merged_df.shape)


print(merged_df.isna().any()[lambda x: x])

print(merged_df.info())

columns_to_fill = ['HU', 'CC']
merged_df[columns_to_fill] = merged_df[columns_to_fill].ffill()

print(merged_df.info())
# Step 4: Display merged result
# print(merged_df.shape)
# print(merged_df.info())

# Updated list of valid features for plotting
valid_features_to_plot = ['TX', 'TN', 'TG', 'RR', 'SS', 'QQ', 'PP', 'HU', 'CC']

# Create subplots
fig, axes = plt.subplots(nrows=len(valid_features_to_plot), ncols=1, figsize=(10, 4 * len(valid_features_to_plot)))

for i, feature in enumerate(valid_features_to_plot):
    axes[i].scatter(merged_df[feature], merged_df['consumption'], alpha=0.5)
    axes[i].set_title(f'Consumption vs {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Consumption')

plt.tight_layout()
plt.show()