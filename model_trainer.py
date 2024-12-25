import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the dataset
def load_dataset(file_name='sensor_data.csv'):
    data = pd.read_csv(file_name)
    return data

# 2. Preprocess the data
def preprocess_data(data):
    # Features: temperature, humidity, sound_volume
    X = data[['temperature', 'humidity', 'sound_volume']]
    
    # Labels: anomaly (0 or 1)
    y = data['anomaly']
    
    # Split the data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 3. Train the Logistic Regression model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# 4. Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("\nModel Evaluation:\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Predict anomalies in new data
def predict_anomalies(model, scaler, new_data):
    # Scale the new data using the same scaler
    new_data_scaled = scaler.transform(new_data)
    
    # Predict anomalies
    predictions = model.predict(new_data_scaled)
    
    # Return predictions (0: Normal, 1: Anomalous)
    return predictions

# 6. Main function to execute the model training and evaluation
def main():
    # Load the dataset
    data = load_dataset('sensor_data.csv')
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Train the anomaly detection model
    model = train_model(X_train, y_train)
    print("Model training complete.")
    
   
   
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Example of using the model to predict new anomalies
    # Let's simulate some new incoming sensor data
    new_data = pd.DataFrame({
        'temperature': [72.3, 98.6, 65.2],    # Example data points
        'humidity': [55.0, 19.2, 48.6],
        'sound_volume': [62.4, 89.7, 61.0]
    })
    
    # Predict anomalies in the new data
    predictions = predict_anomalies(model, scaler, new_data)
    print("\nPredictions for new data (0: Normal, 1: Anomalous):")
    print(predictions)

    # Save the trained model
    joblib.dump(model, 'logistic_regression_model.pkl')

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

# Run the main function
if __name__ == "__main__":
    main()