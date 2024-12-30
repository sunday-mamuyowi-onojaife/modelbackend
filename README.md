IU INTERNATIONAL UNIVERSITY OF APPLIED SCIENCES. 

Project: From Model to Production (DLBDSMTP01). 


Task 1: Anomaly detection in an IoT setting (Spotlight: Stream processing).  
This repository contains the implementation of an anomaly detection system designed for a factory setting. 
The system processes real-time sensor data to identify anomalies in the production cycle for wind turbine components, 
leveraging Python, machine learning, and FastAPI.

Problem Statement.
A cutting-edge modern factory is producing machine components for wind turbines. The factory
is equipped with numerous sensors which track the production cycle. The factory managers can
oversee the numeric values in a BI dashboard. In addition to this, they would like to be
informed by a decision support system when patterns emerge in the sensor data, implying that
something has gone wrong. As the data scientist for this project, you are assigned to develop and
implement an anomaly detection system that integrates with the productive system. You can
rely on the expertise of the shop floor employees who have been working at the production site
for a long time and have gathered valuable domain knowledge. In close consultancy with the
shop floor employees, it has become clear that temperature, humidity, and sound volume are
good indicators for an anomaly in the production cycle and a faulty produced item. You build a
simple machine learning model in Python which uses these features to predict an anomaly
score for each made item. This model should not be very elaborate, but the focus of your task is
to design a model so that it is easily implemented in the productive system. Your system must
be able to process data streams, as the sensors in the factory measure data continuously. Your
implemented model should take these measurements over a standardized API and respond
with a prediction score for an anomaly. Your system must be easily monitorable, maintainable,
scalable, and adaptable to new data.
1. Design the conceptual architecture of your system. By doing so, consider data ingestion,
processing, and handling requests to the prediction model as a service. Draft a visual overview
of your architectural design, showing which data and processes are handed over by which
application to the next. This will also guide you through the next steps of your project.
2. Choose an open data source that can serve as sample data for your project. A good starting
point for your research are open data science competition websites, such as Kaggle.
Alternatively, you can also produce your own fictional sample data. As you have to simulate
continuous streams of data, you basically have three options:
1. Run an application on your local machine which produces a continuous stream of simulated sensor data,
2. Find a continuous open stream of sample data online, or
3. Use a static open data set and simulate its continuous nature by manually updating the data and monitoring the changes in your system
4. Build a simple anomaly detection model using Python. The model should be trained by the
simulated sensor data you acquired in the second step. Do not put too much effort into building
this model, and just make sure that it performs the task at hand and check basic statistical
measures.
5. Package your model in a way that it can take (simulated) sensor data over a standardized
RESTful API and respond with a prediction for an anomaly. You might want to use Python
libraries for this, such as Flask or mlflow. A good starting point for this step is the
documentation of the respective Python libraries. This will be the most work-intensive step of
your project.

Project Overview

This project simulates an IoT environment where real-time sensor data (temperature, humidity, and sound volume)
is monitored for anomalies. An anomaly detection model predicts faulty components, enabling proactive maintenance and minimizing production downtime.

Objectives

Simulate and ingest real-time sensor data.
Build a logistic regression model for anomaly detection.
Deploy the model via a RESTful API using FastAPI.
Display real-time predictions using a frontend interface.

System Conceptual Architecture 

![Architecture](https://github.com/user-attachments/assets/876be19f-f7d5-4a4b-8eb4-1802b7344bb7)
The architecture includes:

Generated Dataset: Simulated sensor data generation in CSV format.
 
Data Stream Simulation: Continuous data streaming to mimic real-time sensor readings.
 
Data Ingestion API: A FastAPI service that receives and processes sensor data.

Stream Processor: A stream processor continuously feeds this data to the Anomaly Detector for real-time analysis

Data Storage (Database/File Storage): Generated sensor data is stored locally in a CSV file format.

Anomaly Detection Model: A logistic regression model to classify normal and abnormal readings.

Anomaly Detector RESTful API: The API, built using FastAPI, processes incoming sensor data and provides a real-time anomaly prediction.

Decision Support System: A frontend interface for user input and real-time anomaly feedback.

Data Generation

The data is synthetically generated to mimic sensor readings in a factory setting. 
The dataset includes 5000 normal and 1600 abnormal instances:
Normal Data: Generated around typical sensor values.
Abnormal Data: Outliers with extreme values to simulate faults.
The data is stored in sensor_data.csv, and a small Python script simulates streaming by printing data at intervals.
Code snippet to generate data:
 generate_dataset.py

Model Training and Deployment
A logistic regression model is trained to distinguish between normal and anomalous data:
Preprocessing: Standard scaling is applied to normalize features.
Training: The model is trained with the generated dataset.
Model Serialization: The trained model and scaler are saved using joblib.
To deploy, FastAPI loads the serialized model, and the /predict endpoint is exposed for anomaly detection requests.
Example code for model training: 
   model.py 
   
API Endpoints
A FastAPI server hosts the model as a RESTful service.
/predict (POST)
Input: JSON with temperature, humidity, and sound_volume.
Output: JSON with prediction (0: normal, 1: anomalous).
Example request:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"temperature\": 30, \"humidity\": 70, \"sound_volume\": 65}"
Code for API deployment: 
uvicorn fastapi:app --reload

Frontend Integration
A React frontend provides a simple interface for entering sensor data and viewing real-time predictions.
It communicates with the FastAPI backend, sending POST requests to /predict and displaying the API's response. 
The frontend setup is managed with Vite, styled using TailwindCSS. 

Installation and Setup

1 Clone the repository: 
*git clone https://github.com/

2 Install dependencies:
pip install -r requirements.txt

3 Generate dataset:
generate_dataset.py

4 Train the model:
model.py

5 Run the FastAPI server backend:

   Open new command prompt, and dont close.

   Directory Desktop 

   cd IU-Model_to_Production 

   uvicorn backend:app  --reload

6 Launch the frontend (React setup): 
   open another new command prompt, and dont close.

   cd Desktop

   cd frontend_iu_model_to_production

   code . , to open vscode 

   at vscode, go to terminal. 

   type npm  run dev 

   highlight the cursor on the localhost and click on the followlink that pop up:   http://localhost:5173/
   a new window will open with the user interface enter temperature 30, humidity 70, sound volume 65.
   Also can check the functionality deployed to cloud: 
       Frontend url: https://sundaymodelfe-a92f16ddabd0.herokuapp.com/ 
       Backend url: https://sundaymodelbe-1c9f72af7300.herokuapp.com/ 

Usage
Once the API server is running, send sensor data to /predict to receive anomaly predictions. 
Example usage: 

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"temperature\": 30, \"humidity\": 70, \"sound_volume\": 65}"

Challenges and Considerations

Data Generation: Simulating realistic sensor data with sufficient variability.

Model Performance: Managing imbalanced data and preventing overfitting to simulated patterns.

API Integration: Ensuring low latency in real-time predictions.

Scalability: Managing API concurrency with Uvicorn and FastAPI.



