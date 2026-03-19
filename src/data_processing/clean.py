import pandas as pd
import logging
import os

def clean_data(input_path="/opt/airflow/data/raw/sensor_data.csv", output_path="/opt/airflow/data/processed/cleaned_data.csv"):
    logging.info("Starting clean_data task...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    df = df[df['vibration'] > 0]
    df = df[df['engine_temperature'] > 0]
    
    df.to_csv(output_path, index=False)
    
    logging.info(f"Cleaned data from {initial_shape} to {df.shape}.")
    logging.info(f"Saved cleaned dataset to {output_path}")
    logging.info("Finished clean_data task.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    clean_data("data/raw/sensor_data.csv", "data/processed/cleaned_data.csv")
