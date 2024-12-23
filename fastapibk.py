
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model for incoming requests
class SensorData(BaseModel):
    temperature: float
    humidity: float
    sound_volume: float

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')  # Assumes the model was saved in this file
scaler = joblib.load('scaler.pkl')  # Assumes the scaler was saved in this file

#home page
@app.get("/")
def index():
    return {"Student Name":"Mamuyowi Onojaife Sunday"}


# Endpoint to predict anomalies
@app.post("/predict")
async def predict_anomaly(data: SensorData):
    # Extract data from request
    sensor_input = np.array([[data.temperature, data.humidity, data.sound_volume]])
    
    # Scale the input data using the pre-trained scaler
    scaled_input = scaler.transform(sensor_input)
    
    # Make prediction using the loaded logistic regression model
    prediction = model.predict(scaled_input)[0]
    
    # Return the prediction (0: normal, 1: anomalous)
    return {"prediction": int(prediction)}

# Example: Running the app
# If you're running this locally, use: uvicorn app:app --reload
