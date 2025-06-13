import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("london_weather_data_1979_to_2023.csv")
df2 = pd.read_csv("london_energy.csv")


print(df.info())



# print(df.isna().any()[lambda x: x])
# df = df.dropna()

print(df.info())


avg_kwh = df2.groupby('Date')['KWH'].mean()
avg_kwh = pd.DataFrame({'date':avg_kwh.index.tolist(), 'consumption':avg_kwh.values.tolist()})
avg_kwh['date'] = pd.to_datetime(avg_kwh['date'])
print(avg_kwh.head())
# Step 2: Convert the weather data 'DATE' column to datetime
# df['date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
# Keep DATE column as datetime (do not convert to string)
df['date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# Merge on 'date' column only
merged_df = pd.merge(avg_kwh, df.drop(columns=['DATE']), on='date', how='inner')



# Step 4: Display merged result

# Output the shape of the merged result for quick verification
print(merged_df.shape)
print(merged_df.head())

