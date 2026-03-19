import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
import joblib

def engineer_features(input_path="/opt/airflow/data/processed/cleaned_data.csv", output_path="/opt/airflow/data/processed/featured_data.csv", scaler_path="/opt/airflow/models/scaler.pkl"):
    logging.info("Starting feature_engineering task...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Rolling features
    df['vibration_rolling_mean_5'] = df['vibration'].rolling(window=5, min_periods=1).mean()
    df['engine_temp_rolling_mean_5'] = df['engine_temperature'].rolling(window=5, min_periods=1).mean()
    
    features = ['vibration', 'engine_temperature', 'pressure', 'fuel_flow', 'altitude', 'rpm', 'vibration_rolling_mean_5', 'engine_temp_rolling_mean_5']
    
    # Scaling
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    joblib.dump(scaler, scaler_path)
    
    df.to_csv(output_path, index=False)
    
    logging.info(f"Engineered features for {len(df)} rows.")
    logging.info(f"Saved featured dataset to {output_path}")
    logging.info("Finished feature_engineering task.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    engineer_features("data/processed/cleaned_data.csv", "data/processed/featured_data.csv", "models/scaler.pkl")
