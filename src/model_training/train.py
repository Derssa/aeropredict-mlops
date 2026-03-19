import pandas as pd
import logging
import os
from sklearn.ensemble import IsolationForest
import joblib

def train_model(input_path="/opt/airflow/data/processed/featured_data.csv", model_path="/opt/airflow/models/anomaly_model.pkl"):
    logging.info("Starting train_model task...")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    df = pd.read_csv(input_path)
    features = ['vibration', 'engine_temperature', 'pressure', 'fuel_flow', 'altitude', 'rpm', 'vibration_rolling_mean_5', 'engine_temp_rolling_mean_5']
    X = df[features]
    
    logging.info(f"Training IsolationForest on {len(X)} samples with {len(features)} features...")
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)
    
    joblib.dump(model, model_path)
    
    logging.info(f"Saved trained model to {model_path}")
    logging.info("Finished train_model task.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_model("data/processed/featured_data.csv", "models/anomaly_model.pkl")
