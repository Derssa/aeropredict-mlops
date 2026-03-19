from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="AeroPredict Anomaly Detection API")

# Define the structure of the incoming sensor data
class SensorData(BaseModel):
    vibration: float
    engine_temperature: float
    pressure: float
    fuel_flow: float
    altitude: float
    rpm: float
    # We will simulate the rolling means for simplicity in a single request, 
    # but in a real system, the API or a stream processor would calculate these over time.
    vibration_rolling_mean_5: float
    engine_temp_rolling_mean_5: float

# Load the model and scaler when the API starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/production_anomaly_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../models/scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    model = None
    scaler = None
    print("Warning: Model or scaler not found. Please run the Airflow pipeline first.")

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AeroPredict Live Testing</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
            .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
            input { background: #1e293b; border: 1px solid #334155; color: white; padding: 0.5rem; border-radius: 0.375rem; width: 100%; transition: 0.3s; }
            input:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5); }
        </style>
    </head>
    <body class="min-h-screen flex items-center justify-center p-6">
        <div class="glass max-w-4xl w-full rounded-2xl p-8 shadow-2xl">
            <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-400 mb-2">AeroPredict ML</h1>
            <p class="text-slate-400 mb-8">Live Aircraft Sensor Anomaly Testing Dashboard</p>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Form -->
                <div class="space-y-4">
                    <h2 class="text-xl font-semibold mb-4 text-blue-300">Sensor Telemetry</h2>
                    <form id="sensorForm" class="space-y-4">
                        <div class="grid grid-cols-2 gap-4">
                            <div><label class="text-xs text-slate-400">Vibration</label><input type="number" step="0.1" id="vibration" value="5.2"></div>
                            <div><label class="text-xs text-slate-400">Engine Temp (°C)</label><input type="number" step="0.1" id="engine_temperature" value="810.0"></div>
                            <div><label class="text-xs text-slate-400">Pressure</label><input type="number" step="0.1" id="pressure" value="31.0"></div>
                            <div><label class="text-xs text-slate-400">Fuel Flow</label><input type="number" step="0.1" id="fuel_flow" value="1205.0"></div>
                            <div><label class="text-xs text-slate-400">Altitude (ft)</label><input type="number" step="0.1" id="altitude" value="35000.0"></div>
                            <div><label class="text-xs text-slate-400">RPM</label><input type="number" step="0.1" id="rpm" value="10050.0"></div>
                            <div><label class="text-xs text-slate-400">Vib. Rolling (5s)</label><input type="number" step="0.1" id="vibration_rolling_mean_5" value="5.1"></div>
                            <div><label class="text-xs text-slate-400">Temp. Rolling (5s)</label><input type="number" step="0.1" id="engine_temp_rolling_mean_5" value="809.0"></div>
                        </div>
                        <div class="flex space-x-4 mt-6">
                            <button type="submit" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded-lg transition shadow-lg">Run Prediction</button>
                            <button type="button" onclick="loadAnomaly()" class="w-full bg-red-600 hover:bg-red-500 text-white font-bold py-3 px-4 rounded-lg transition shadow-lg" title="Loads data mimicking an engine breakdown">Simulate Breakdown</button>
                        </div>
                    </form>
                </div>

                <!-- Result -->
                <div class="flex flex-col justify-center border-l-0 md:border-l border-slate-700 md:pl-8 pt-6 md:pt-0">
                    <h2 class="text-xl font-semibold mb-4 text-slate-300">AI Analysis</h2>
                    <div id="resultBox" class="rounded-xl p-6 bg-slate-800/50 border border-slate-700 h-56 flex flex-col justify-center items-center text-center transition-all duration-500">
                        <span class="text-slate-400">Awaiting Telemetry...</span>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function loadAnomaly() {
                document.getElementById('vibration').value = "25.2";
                document.getElementById('engine_temperature').value = "950.0";
                document.getElementById('vibration_rolling_mean_5').value = "25.1";
                document.getElementById('engine_temp_rolling_mean_5').value = "910.0";
                document.getElementById('sensorForm').dispatchEvent(new Event('submit'));
            }

            document.getElementById('sensorForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const resultBox = document.getElementById('resultBox');
                resultBox.innerHTML = '<span class="text-slate-400 animate-pulse">Running AI Model...</span>';
                
                const data = {
                    vibration: parseFloat(document.getElementById('vibration').value),
                    engine_temperature: parseFloat(document.getElementById('engine_temperature').value),
                    pressure: parseFloat(document.getElementById('pressure').value),
                    fuel_flow: parseFloat(document.getElementById('fuel_flow').value),
                    altitude: parseFloat(document.getElementById('altitude').value),
                    rpm: parseFloat(document.getElementById('rpm').value),
                    vibration_rolling_mean_5: parseFloat(document.getElementById('vibration_rolling_mean_5').value),
                    engine_temp_rolling_mean_5: parseFloat(document.getElementById('engine_temp_rolling_mean_5').value)
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if(result.is_anomaly) {
                         resultBox.className = "rounded-xl p-6 bg-red-900/40 border border-red-500 h-56 flex flex-col justify-center items-center text-center transition-all duration-500 shadow-[0_0_30px_rgba(239,68,68,0.3)]";
                         resultBox.innerHTML = `
                             <svg class="w-12 h-12 text-red-500 mb-2 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                             <h3 class="text-2xl font-bold text-red-400">CRITICAL ANOMALY</h3>
                             <p class="text-red-200 mt-2">` + result.status + `</p>
                             <span class="text-xs text-red-400 mt-4 font-mono">Score: ` + result.anomaly_score.toFixed(4) + `</span>
                         `;
                    } else {
                         resultBox.className = "rounded-xl p-6 bg-emerald-900/40 border border-emerald-500 h-56 flex flex-col justify-center items-center text-center transition-all duration-500 shadow-[0_0_30px_rgba(16,185,129,0.2)]";
                         resultBox.innerHTML = `
                             <svg class="w-12 h-12 text-emerald-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                             <h3 class="text-2xl font-bold text-emerald-400">SYSTEM NORMAL</h3>
                             <p class="text-emerald-200 mt-2">` + result.status + `</p>
                             <span class="text-xs text-emerald-400 mt-4 font-mono">Score: ` + result.anomaly_score.toFixed(4) + `</span>
                         `;
                    }
                } catch (error) {
                    resultBox.innerHTML = '<span class="text-red-500">Error connecting to server.</span>';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
def predict_anomaly(data: SensorData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not trained yet. Run the Airflow pipeline.")

    # Convert the incoming JSON data into a format that our model understands (Pandas DataFrame)
    features = ['vibration', 'engine_temperature', 'pressure', 'fuel_flow', 'altitude', 'rpm', 'vibration_rolling_mean_5', 'engine_temp_rolling_mean_5']
    
    input_df = pd.DataFrame([data.dict()])
    
    # Scale the features using the scaler we saved during training
    input_df[features] = scaler.transform(input_df[features])
    
    # Predict (-1 is an anomaly, 1 is normal for IsolationForest)
    prediction = model.predict(input_df)[0]
    score = model.decision_function(input_df)[0]
    
    is_anomaly = True if prediction == -1 else False
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": float(score),
        "status": "URGENT ALARM! Engine failure likely." if is_anomaly else "Normal flight."
    }
