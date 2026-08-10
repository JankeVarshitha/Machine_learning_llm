import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

def get_tabular_scarcity_scenario():
    """
    SCENARIO 1: Extreme Cold-Start / Data Scarcity
    Simulates launching a product with only 15 historical data rows.
    """
    print(" Loading California Housing & Injecting Extreme Data Scarcity...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    median_val = df['MedHouseVal'].median()
    df['Target'] = (df['MedHouseVal'] > median_val).astype(int)
    df = df.drop(columns=['MedHouseVal'])
    
    df_sample = df.sample(n=200, random_state=42)
    
    # CRITICAL: Train on only 15 rows to force over-fitting on traditional trees
    train_df = df_sample.iloc[:15] 
    test_df = df_sample.iloc[15:]
    
    return train_df.drop(columns=['Target']), train_df['Target'], test_df.drop(columns=['Target']), test_df['Target']

def get_timeseries_drift_scenario():
    """
    SCENARIO 2: Abrupt Structural Concept Drift
    Simulates a sudden system failure or severe macroeconomic shift.
    """
    print(" Generating Multi-Seasonal Time-Series with Sudden Concept Drift...")
    np.random.seed(42)
    time_steps = np.arange(0, 700)
    
    # Base multi-seasonal pattern
    signal = 30 + (0.02 * time_steps) + 10 * np.sin(time_steps / 24) + 5 * np.cos(time_steps / 168)
    
    # Inject severe Concept Drift in the last 48 hours (the test horizon)
    # Sudden drop in baseline, 4x noise multiplier, and inverted seasonality
    drift_mask = time_steps >= (700 - 48)
    signal[drift_mask] = signal[drift_mask] - 25  
    
    noise = np.random.normal(0, 1.5, size=len(time_steps))
    noise[drift_mask] = noise[drift_mask] * 4.0 
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=700, freq="H"),
        "value": signal + noise
    })
    return df