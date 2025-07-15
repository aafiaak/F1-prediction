import os
import fastf1
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("f1cache", exist_ok=True)
fastf1.Cache.enable_cache("f1cache")
fastf1.set_log_level("ERROR")

conn = sqlite3.connect("f1data.db")

def store_data_in_db(df, table_name, conn):
    df.to_sql(table_name, conn, if_exists="replace", index=False)

# Race data collected from 2022-2024 Bahrain GPs

years = [2022, 2023, 2024]
race_laps = []

for year in years:
    session = fastf1.get_session(year, "Bahrain Grand Prix", "R")
    session.load()
    laps = session.laps[["Driver", "LapTime"]].dropna(subset=["LapTime"]).copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Year"] = year
    race_laps.append(laps)

all_race_laps = pd.concat(race_laps, ignore_index=True)
store_data_in_db(all_race_laps, "race_laps", conn)
avg24 = all_race_laps.groupby("Driver", as_index=False)["LapTimeSec"].mean()

quali25 = fastf1.get_session(2025, "Bahrain Grand Prix", "Q")
quali25.load()
q_laps = (
    quali25.laps[["Driver", "LapTime"]]
    .dropna(subset=["LapTime"])
    .copy()
)
q_laps["QualiTimeSec"] = q_laps["LapTime"].dt.total_seconds()
best_q25 = q_laps.groupby("Driver", as_index=False)["QualiTimeSec"].min()
best_q25["QualiRank"] = best_q25["QualiTimeSec"].rank()
store_data_in_db(best_q25, "qualifying_laps", conn)
# avg24 = laps24.groupby("Driver", as_index=False)["LapTimeSec"].mean()

df_train = pd.merge(best_q25, avg24, on="Driver")
X = df_train[["QualiTimeSec", "QualiRank"]]
y = df_train["LapTimeSec"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# To see performance
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMAE on hold out: {mae:.3f} s")

# To see our predicted lap time versus actual lap time
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Actual versus Predicted Lap Times")
plt.grid(True)
plt.tight_layout()
plt.show()

# Prediction
df_train["PredLapSec"] = model.predict(df_train[["QualiTimeSec", "QualiRank"]])
top5 = df_train.nsmallest(5, "PredLapSec")
print("\nPredicted Top 5 for 2025 Bahrain GP")
print(top5[["Driver", "PredLapSec"]].to_string(index=False))

# Compare against actual 2025 results
race_2025 = fastf1.get_session(2025, "Bahrain Grand Prix", "R")
race_2025.load()
results_25 = race_2025.results
actual_top5 = results_25.nsmallest(5, "Position")["Abbreviation"].tolist()
predicted_top5 = top5["Driver"].tolist()

correct_matches = len(set(predicted_top5) & set(actual_top5))
accuracy = correct_matches / 5 * 100
print(f"\nModel Top 5 Match Accuraccy: {accuracy:.1f}%")
print(f"Actual Top 5:   {actual_top5}")
print(f"Predicted Top 5: {predicted_top5}")

conn.close()
