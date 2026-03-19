import logging
import shutil
import os

def save_model(source_path="/opt/airflow/models/anomaly_model.pkl", dest_path="/opt/airflow/models/production_anomaly_model.pkl"):
    logging.info("Starting save_model task...")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        logging.info(f"Model successfully saved to {dest_path}")
    else:
        logging.error(f"Source model not found at {source_path}")
    
    logging.info("Finished save_model task.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    save_model()
