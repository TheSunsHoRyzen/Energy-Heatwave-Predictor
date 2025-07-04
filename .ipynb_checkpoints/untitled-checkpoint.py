import pandas as pd
import glob
import os

# 1) Grab every historic_demand CSV under archive/
energy_files = glob.glob(os.path.join("archive", "historic_demand_*.csv"))

energy_dfs = []
for fp in energy_files:
    df = pd.read_csv(fp)
    # Parse the true date column
    df["date"] = pd.to_datetime(df["SETTLEMENT_DATE"], format="%d-%b-%Y")
    # Compute a single daily consumption metric (mean or sum)
    daily = (
        df
        .groupby("date")["ENGLAND_WALES_DEMAND"]
        .mean()                               # or .sum()
        .reset_index(name="consumption")
    )
    energy_dfs.append(daily)

# 2) Concatenate them into one DataFrame
energy_all = pd.concat(energy_dfs, ignore_index=True).sort_values("date")
print("Energy table:", energy_all.shape)
energy_all.head()
# 1) Find the weather files
weather_files = glob.glob(os.path.join("archive", "uk_weather*.csv"))

weather_dfs = []
for fp in weather_files:
    w = pd.read_csv(fp)
    # Parse its date column—adjust format if needed
    w["date"] = pd.to_datetime(w["DATE"], format="%Y%m%d")
    weather_dfs.append(w)

weather_all = pd.concat(weather_dfs, ignore_index=True).sort_values("date")
print("Weather table:", weather_all.shape)
weather_all.head()
merged_uk = pd.merge(
    energy_all,
    weather_all.drop(columns=["DATE"]),  # drop the raw DATE col if you like
    on="date",
    how="inner"
)
print("Merged UK dataset:", merged_uk.shape)
merged_uk.head()
