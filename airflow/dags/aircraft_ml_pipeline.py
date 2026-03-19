from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ensure Airflow can find our src modules since PYTHONPATH might not be perfectly set in all contexts
sys.path.append('/opt/airflow')

from src.data_generation.generate import generate_sensor_data
from src.data_processing.clean import clean_data
from src.feature_engineering.engineer import engineer_features
from src.model_training.train import train_model
from src.model_evaluation.evaluate import evaluate_model
from src.model_training.save import save_model

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='aircraft_ml_pipeline',
    default_args=default_args,
    description='AeroPredict MLOps - Aircraft Sensor Anomaly Detection Pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'aeropredict']
) as dag:

    # Define tasks
    t1_generate = PythonOperator(
        task_id='generate_sensor_data',
        python_callable=generate_sensor_data,
        op_kwargs={'output_path': '/opt/airflow/data/raw/sensor_data.csv', 'n_rows': 50000}
    )

    t2_clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        op_kwargs={
            'input_path': '/opt/airflow/data/raw/sensor_data.csv',
            'output_path': '/opt/airflow/data/processed/cleaned_data.csv'
        }
    )

    t3_engineer = PythonOperator(
        task_id='feature_engineering',
        python_callable=engineer_features,
        op_kwargs={
            'input_path': '/opt/airflow/data/processed/cleaned_data.csv',
            'output_path': '/opt/airflow/data/processed/featured_data.csv',
            'scaler_path': '/opt/airflow/models/scaler.pkl'
        }
    )

    t4_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={
            'input_path': '/opt/airflow/data/processed/featured_data.csv',
            'model_path': '/opt/airflow/models/anomaly_model.pkl'
        }
    )

    t5_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        op_kwargs={
            'input_path': '/opt/airflow/data/processed/featured_data.csv',
            'model_path': '/opt/airflow/models/anomaly_model.pkl',
            'output_path': '/opt/airflow/data/processed/predictions.csv'
        }
    )

    t6_save = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
        op_kwargs={
            'source_path': '/opt/airflow/models/anomaly_model.pkl',
            'dest_path': '/opt/airflow/models/production_anomaly_model.pkl'
        }
    )

    # Define dependencies
    t1_generate >> t2_clean >> t3_engineer >> t4_train >> t5_evaluate >> t6_save
