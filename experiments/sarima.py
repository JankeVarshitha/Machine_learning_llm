import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Updated import
import timesfm
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

print("\n=== STARTING 720-HOUR (30-DAY) AEP ENERGY BENCHMARK WITH SARIMA ===")

# --- STEP 1: LOAD REAL AEP ENERGY CONSUMPTION DATA ---
df_raw = pd.read_csv("AEP_hourly.csv")
df_raw['timestamp'] = pd.to_datetime(df_raw['Datetime'])
df_raw['value'] = df_raw['AEP_MW'].astype(np.float32)
df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

# --- STEP 2: CLEAN UNIFORM COHORT ALIGNMENT ---
df = df_raw[['timestamp', 'value']].copy()
# Expand lags to give LightGBM some broader history
for lag in range(1, 25):
    df[f"lag_{lag}"] = df["value"].shift(lag)
df.dropna(inplace=True)

# STRESS TEST EXTENSION: 720 Hours (30 Full Days)
horizon = 720
train_df, test_df = df.iloc[:-horizon], df.iloc[-horizon:]

X_train, y_train = train_df.drop(columns=["timestamp", "value"]), train_df["value"]
X_test, y_test = test_df.drop(columns=["timestamp", "value"]), test_df["value"]

y_train_raw = y_train.values.astype(np.float32)


# --- MODEL 1: LIGHTGBM (CORRECT RECURSIVE MULTI-STEP GENERATION) ---
t0 = time.time()
lgb_reg = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1).fit(X_train, y_train)

lgb_pred = []
feature_names = X_train.columns.tolist()

# Extract the 1D values array from the FIRST row of the test feature block using bracket notation
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

    # Explicitly assign the flat scalar prediction into the lag_1 index position
    new_features[0] = pred_step

    # Progress the row vector array for the next loop tick
    current_features_array = new_features

lgb_pred = np.array(lgb_pred)
lgb_latency = (time.time() - t0) * 1000


# --- MODEL 2: HF TIMESFM 2.5 (NATIVE 720-STEP GENERATION) ---
t1 = time.time()
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=2048,
        max_horizon=horizon, 
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        fix_quantile_crossing=True
    )
)

point_forecast, _ = model.forecast(
    horizon=horizon,
    inputs=[y_train_raw]
)
timesfm_pred = point_forecast[0, :horizon]
timesfm_latency = (time.time() - t1) * 1000


# --- MODEL 3: SARIMA (REPLACES FLATTENING ARIMA) ---
t2 = time.time()

print("Fitting optimized low-memory SARIMA on full dataset...")

# --- FIXED MODEL 3: LOW-MEMORY, DRIFT-LOCKED SARIMA ---
# Using pm.ARIMA instead of statsmodels drastically reduces memory allocation overhead
sarima_model = pm.ARIMA(
    order=(1, 0, 1), 
    seasonal_order=(1, 0, 1, 24),
    simple_differencing=True, 
    with_intercept=False,                        # 'c' forces a stable constant baseline, preventing the upward explosion
    suppress_warnings=True,
    error_action='ignore'
).fit(y_train.values)                    # <-- NOW RUNNING ON THE FULL RAW TRAINING VALUES!

sarima_pred = sarima_model.predict(n_periods=horizon)
sarima_latency = (time.time() - t2) * 1000

print(f"SARIMA finished on full history in {sarima_latency:.1f} ms")


# --- TERMINAL PERFORMANCE METRICS LOGGING ---
print(f"\n{'Model':<12} | {'MAE':<10} | {'RMSE':<10} | {'Latency':<10}")
print("-" * 50)
print(f"LightGBM     | {mean_absolute_error(y_test, lgb_pred):<10.2f} | {np.sqrt(mean_squared_error(y_test, lgb_pred)):<10.2f} | {lgb_latency:.1f} ms")
print(f"TimesFM 2.5  | {mean_absolute_error(y_test, timesfm_pred):<10.2f} | {np.sqrt(mean_squared_error(y_test, timesfm_pred)):<10.2f} | {timesfm_latency:.1f} ms")
print(f"SARIMA       | {mean_absolute_error(y_test, sarima_pred):<10.2f} | {np.sqrt(mean_squared_error(y_test, sarima_pred)):<10.2f} | {sarima_latency:.1f} ms")

# --- STEP 3: LONG-RANGE PLOT GENERATION ---
plt.figure(figsize=(14, 5))
plt.plot(y_test.values, label="Truth (Actual AEP MW)", color="black", linewidth=2)
plt.plot(lgb_pred, label="LightGBM (Recursive)", color="cyan", linestyle="--")
plt.plot(timesfm_pred, label="TimesFM 2.5 (Zero-Shot)", color="magenta", linestyle=":")
plt.plot(sarima_pred, label="SARIMA (24h Seasonal)", color="green", linestyle="-.")

# Draw vertical lines to mark day updates on the 720h canvas
for day in range(24, horizon, 24):
    plt.axvline(x=day, color='gray', linestyle=':', alpha=0.3)
    # Label every 5th day to avoid visual cluttering
    if (day // 24) % 5 == 0 or day // 24 == 1:
        plt.text(day - 12, plt.ylim()[1] * 0.97, f"Day {day//24}", color='gray', fontsize=8, ha='center')

plt.title("Energy Consumption: 30-Day (720-Hour) Forecast Comparison for different models")
plt.xlabel("Forecast Horizon Steps (Hours)")
plt.ylabel("Megawatts (MW)")
plt.legend(loc="upper right")
plt.tight_layout()

plt.savefig("aep_720h_stress_test_fix_zero_diff.png")
print("\n=== LONG-RANGE BENCHMARK COMPLETE: SAVED TO 'aep_720h_stress_test_fix_zero_diff.png' ===")
