import pandas as pd
import logging
import os
import joblib

def evaluate_model(input_path="/opt/airflow/data/processed/featured_data.csv", model_path="/opt/airflow/models/anomaly_model.pkl", output_path="/opt/airflow/data/processed/predictions.csv"):
    logging.info("Starting evaluate_model task...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(input_path)
    features = ['vibration', 'engine_temperature', 'pressure', 'fuel_flow', 'altitude', 'rpm', 'vibration_rolling_mean_5', 'engine_temp_rolling_mean_5']
    X = df[features]
    
    model = joblib.load(model_path)
    
    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly'] = model.predict(X)  # -1 for anomaly, 1 for normal
    
    num_anomalies = (df['is_anomaly'] == -1).sum()
    anomaly_rate = num_anomalies / len(df)
    
    logging.info(f"Evaluation Metrics:")
    logging.info(f"Total samples evaluated: {len(df)}")
    logging.info(f"Total anomalies detected: {num_anomalies}")
    logging.info(f"Anomaly rate: {anomaly_rate:.4f}")
    
    df.to_csv(output_path, index=False)
    logging.info(f"Saved predictions to {output_path}")
    logging.info("Finished evaluate_model task.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    evaluate_model("data/processed/featured_data.csv", "models/anomaly_model.pkl", "data/processed/predictions.csv")
