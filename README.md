# AeroPredict MLOps – Aircraft Sensor Anomaly Detection Pipeline

## Project Overview

AeroPredict MLOps is an end-to-end Machine Learning pipeline project designed to simulate an aircraft predictive maintenance monitoring system. The project builds an automated anomaly detection ML system using Apache Airflow to orchestrate data generation, cleaning, feature engineering, model training, model evaluation, and saving. Furthermore, it utilizes an embedded FastAPI application with a live UI dashboard to instantly test real-time aircraft sensory data against the trained model algorithm to predict mechanical failure probability.
<img width="1898" height="963" alt="Screenshot 2026-03-18 205959" src="https://github.com/user-attachments/assets/4e686bb1-a7ad-4d5a-9496-b0ede59c45ab" />

## System Architecture

The project is completely containerized using Docker and Docker Compose. It leverages:
- **Apache Airflow**: For orchestrating the ML workflow.
- **PostgreSQL**: Serving as the Airflow metadata database.
- **Redis**: For managing the task queue (CeleryExecutor).
- **Python Data Stack**: `pandas`, `numpy`, `scikit-learn` for generating synthetic sensor data, processing it, and training an `IsolationForest` anomaly detection model.
- **FastAPI**: Running a self-hosted API and User Interface to test prediction capabilities of the built model.
- **Docker**: For consistent, reproducible environment setup.

The pipeline stages are organized strictly into separated Python modules in the `src` folder.

## Project Structure

```text
aeropredict-mlops/
├── docker-compose.yml       # Docker Compose configuration for Airflow setup
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── airflow/
│   └── dags/                # Airflow Directed Acyclic Graphs
│       └── aircraft_ml_pipeline.py
├── src/
│   ├── api/                 # FastAPI configuration serving the HTML dashboard and predict logic
│   ├── data_generation/     # Generation of synthetic aircraft sensor data
│   ├── data_processing/     # Data cleaning and handling missing values
│   ├── feature_engineering/ # Creating features for the ML model
│   ├── model_training/      # Training the Isolation Forest model
│   └── model_evaluation/    # Evaluating model anomaly scores
├── models/                  # Saved model artifacts (e.g., anomaly_model.pkl)
└── data/
    ├── raw/                 # Raw simulated sensor data
    └── processed/           # Processed and engineered datasets
```

## 1. How to Start the Airflow Pipeline

1. Make sure you have Docker and Docker Compose installed.
2. Navigate to the project directory:
   ```bash
   cd aeropredict-mlops
   ```
3. Start the Docker containers:
   ```bash
   docker compose up -d
   ```
   *(Note: On the first run, the `airflow-init` container will configure the database and create the initial directories. The other containers will wait for it to finish.)*

## 2. Using the Airflow UI

Once all services are healthy and running:
1. Open your web browser and navigate to: [http://localhost:8080](http://localhost:8080)
2. **Login Credentials**:
   - **Username**: airflow
   - **Password**: airflow
3. Turn on and trigger the `aircraft_ml_pipeline` DAG. This will execute the machine learning workflow, creating the required `models/production_anomaly_model.pkl` and `models/scaler.pkl` files.

**Pipeline Breakdown:**
- **`generate_sensor_data`**: Generates 50,000 rows of synthetic aircraft telemetry including timestamp, vibration, engine temperature, pressure, fuel flow, altitude, and RPM. Output is saved to `data/raw/sensor_data.csv`.
- **`clean_data`**: Reads the raw data, handles any anomalies or missing records, and saves the cleaned dataset.
- **`feature_engineering`**: Applies transformations such as standard scaling to prepare for the model.
- **`train_model`**: Trains a Scikit-Learn `IsolationForest` to detect anomalies based on the engineered features.
- **`evaluate_model`**: Uses the trained model to predict anomalies and logs metrics like the proportion of detected anomalies.
- **`save_model`**: Persists the successfully trained model as a `.pkl` file for the API.

## 3. How to Launch the Live API Testing Dashboard

Once the Airflow pipeline has run completely and generated the model artifacts locally in the `models/` directory, you can launch the model testing API.

1. Ensure prerequisites are installed natively in your Python environment:
   ```bash
   pip install -r requirements.txt
   ```
   Or specifically: `pip install fastapi uvicorn pandas scikit-learn numpy joblib`

2. Start the Uvicorn webserver:
   ```bash
   uvicorn src.api.main:app --reload
   ```

3. Open your browser to test live inference visually via the hosted dashboard:
   **[http://localhost:8000/](http://localhost:8000/)**
<img width="1917" height="967" alt="Screenshot 2026-03-18 210117" src="https://github.com/user-attachments/assets/030d5b4e-4ded-45ae-a23e-2bec6a63c2cd" />
<img width="1918" height="957" alt="Screenshot 2026-03-18 210050" src="https://github.com/user-attachments/assets/c14c823a-8603-484c-b849-b96ab4eb0950" />

To interact through a backend API endpoint, simply send a POST JSON body to `http://localhost:8000/predict`.
