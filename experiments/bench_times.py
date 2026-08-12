import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
import timesfm

print("\n=== STARTING 168-HOUR (7-DAY) AEP ENERGY BENCHMARK ===")

# --- STEP 1: LOAD REAL AEP ENERGY CONSUMPTION DATA ---
try:
    df_raw = pd.read_csv("AEP_hourly.csv")
    df_raw['timestamp'] = pd.to_datetime(df_raw['Datetime'])
    df_raw['value'] = df_raw['AEP_MW'].astype(np.float32)
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
except FileNotFoundError:
    print("Error: 'AEP_hourly.csv' not found. Creating seasonal backup mockup for validation.")
    times = pd.date_range(start="2023-01-01", periods=2000, freq="h")
    # Generate daily and weekly multi-seasonal patterns
    values = 15000 + 3000 * np.sin(2 * np.pi * times.hour / 24) + 1500 * np.sin(2 * np.pi * times.dayofweek / 7)
    df_raw = pd.DataFrame({"timestamp": times, "value": values.astype(np.float32)})

# --- STEP 2: CLEAN UNIFORM COHORT ALIGNMENT ---
df = df_raw[['timestamp', 'value']].copy()
# Expand lags to give LightGBM some broader history
for lag in range(1, 25): 
    df[f"lag_{lag}"] = df["value"].shift(lag)
df.dropna(inplace=True)

# STRESS TEST EXTENSION: 168 Hours (7 Full Days)
horizon = 168
train_df, test_df = df.iloc[:-horizon], df.iloc[-horizon:]

X_train, y_train = train_df.drop(columns=["timestamp", "value"]), train_df["value"]
X_test, y_test = test_df.drop(columns=["timestamp", "value"]), test_df["value"]

y_train_raw = y_train.values.astype(np.float32)


# --- MODEL 1: LIGHTGBM (CORRECT RECURSIVE MULTI-STEP GENERATION) ---
t0 = time.time()
lgb_reg = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1).fit(X_train, y_train)

lgb_pred = []
feature_names = X_train.columns.tolist()

# CRITICAL FIX: Extract the 1D values array from the FIRST row of the test feature block using bracket notation
current_features_array = X_test.iloc[0].values.copy()

for i in range(horizon):
    # Convert the 1D numpy row state array into a pandas row matching your column tags
    input_df = pd.DataFrame([current_features_array], columns=feature_names)
    
    # Predict the single next hour step sequence scalar
    pred_step = lgb_reg.predict(input_df)[0]
    lgb_pred.append(pred_step)
    
    # RECURSIVE LAG SHIFT: Create an active copy buffer to propagate historical metrics forward
    new_features = np.zeros_like(current_features_array)
    
    # Shift older lags downstream: lag_2 becomes what lag_1 was, lag_3 becomes lag_2, etc.
    new_features[1:] = current_features_array[:-1] 
    
    # CRITICAL FIX: Explicitly assign the flat scalar prediction into the lag_1 index position
    new_features[0] = pred_step                    
    
    # Progress the row vector array for the next loop tick
    current_features_array = new_features

lgb_pred = np.array(lgb_pred)
lgb_latency = (time.time() - t0) * 1000




# --- MODEL 2: HF TIMESFM 2.5 (NATIVE 168-STEP GENERATION) ---
t1 = time.time()
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=2048,
        max_horizon=horizon,  # Recompiled for 168 steps
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True
    )
)

point_forecast, _ = model.forecast(
    horizon=horizon,
    inputs=[y_train_raw]
)
timesfm_pred = point_forecast[0, :horizon]
timesfm_latency = (time.time() - t1) * 1000

# --- MODEL 3: ARIMA ---
t2 = time.time()
arima_model = ARIMA(y_train_raw, order=(2, 1, 2)).fit()
arima_pred = arima_model.forecast(steps=horizon)
arima_latency = (time.time() - t2) * 1000

# --- TERMINAL PERFORMANCE METRICS LOGGING ---
print(f"\n{"Model":<12} | {"MAE":<10} | {"RMSE":<10} | {"Latency":<10}")
print("-" * 50)
print(f"LightGBM     | {mean_absolute_error(y_test, lgb_pred):<10.2f} | {np.sqrt(mean_squared_error(y_test, lgb_pred)):<10.2f} | {lgb_latency:.1f} ms")
print(f"TimesFM 2.5  | {mean_absolute_error(y_test, timesfm_pred):<10.2f} | {np.sqrt(mean_squared_error(y_test, timesfm_pred)):<10.2f} | {timesfm_latency:.1f} ms")
print(f"ARIMA        | {mean_absolute_error(y_test, arima_pred):<10.2f} | {np.sqrt(mean_squared_error(y_test, arima_pred)):<10.2f} | {arima_latency:.1f} ms")

# --- STEP 3: LONG-RANGE PLOT GENERATION ---
plt.figure(figsize=(14, 5))
plt.plot(y_test.values, label="Truth (Actual AEP MW)", color="black", linewidth=2)
plt.plot(lgb_pred, label="LightGBM (Recursive)", color="cyan", linestyle="--")
plt.plot(timesfm_pred, label="TimesFM 2.5 (Zero-Shot)", color="magenta", linestyle=":")
plt.plot(arima_pred, label="ARIMA", color="green", linestyle="-.")

# Draw vertical lines to mark day updates on the 168h canvas
for day in range(24, horizon, 24):
    plt.axvline(x=day, color='gray', linestyle=':', alpha=0.5)
    plt.text(day - 12, plt.ylim()[1] * 0.97, f"Day {day//24}", color='gray', fontsize=9, ha='center')

plt.title("AEP Energy Consumption: 7-Day (168-Hour) Stress-Test Forecast Comparison")
plt.xlabel("Forecast Horizon Steps (Hours)")
plt.ylabel("Megawatts (MW)")
plt.legend(loc="upper right")
plt.tight_layout()

plt.savefig("aep_168h_stress_test.png")
print("\n=== LONG-RANGE BENCHMARK COMPLETE: SAVED TO 'aep_168h_stress_test.png' ===")
