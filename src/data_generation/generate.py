import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os

def generate_sensor_data(output_path="/opt/airflow/data/raw/sensor_data.csv", n_rows=50000):
    logging.info("Starting generate_sensor_data task...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_time = datetime.now()
    timestamps = [start_time - timedelta(minutes=i) for i in range(n_rows)][::-1]
    
    np.random.seed(42)
    vibration = np.random.normal(loc=5.0, scale=0.5, size=n_rows)
    engine_temp = np.random.normal(loc=800.0, scale=20.0, size=n_rows)
    pressure = np.random.normal(loc=30.0, scale=2.0, size=n_rows)
    fuel_flow = np.random.normal(loc=1200.0, scale=50.0, size=n_rows)
    altitude = np.random.normal(loc=35000.0, scale=100.0, size=n_rows)
    rpm = np.random.normal(loc=10000.0, scale=200.0, size=n_rows)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vibration': vibration,
        'engine_temperature': engine_temp,
        'pressure': pressure,
        'fuel_flow': fuel_flow,
        'altitude': altitude,
        'rpm': rpm
    })
    
    # Inject anomalies
    anomaly_indices = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df.loc[anomaly_indices, 'vibration'] = df.loc[anomaly_indices, 'vibration'] * np.random.uniform(2, 4, size=len(anomaly_indices))
    df.loc[anomaly_indices, 'engine_temperature'] = df.loc[anomaly_indices, 'engine_temperature'] + np.random.uniform(50, 150, size=len(anomaly_indices))
    
    df.to_csv(output_path, index=False)
    
    logging.info(f"Generated {len(df)} rows of synthetic sensor data.")
    logging.info(f"Saved dataset to {output_path}")
    logging.info("Finished generate_sensor_data task.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generate_sensor_data("data/raw/sensor_data.csv")
