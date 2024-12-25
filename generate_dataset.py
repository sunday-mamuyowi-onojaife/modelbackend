import numpy as np
import pandas as pd
import time
import os
# 1. Generate Synthetic Sensor Data
def generate_sensor_data(n_normal=5000, n_abnormal=1600, save_to_file=True, file_name='sensor_data.csv'):
    np.random.seed(42)  # Set seed for reproducibility
    
    # Normal sensor data
    normal_temperature = np.random.normal(loc=70, scale=5, size=n_normal)   # Normal temperature: Mean 70°F, std dev 5
    normal_humidity = np.random.normal(loc=50, scale=10, size=n_normal)     # Normal humidity: Mean 50%, std dev 10
    normal_sound = np.random.normal(loc=60, scale=8, size=n_normal)         # Normal sound: Mean 60dB, std dev 8
    
    # Abnormal sensor data
    abnormal_temperature = np.random.normal(loc=100, scale=5, size=n_abnormal)  # Abnormally high temperature
    abnormal_humidity = np.random.normal(loc=20, scale=5, size=n_abnormal)      # Abnormally low humidity
    abnormal_sound = np.random.normal(loc=90, scale=5, size=n_abnormal)         # Abnormally high sound volume
    
    # Labels for normal and abnormal data
    normal_labels = np.zeros(n_normal)  # Label as 0 for normal
    abnormal_labels = np.ones(n_abnormal)  # Label as 1 for abnormal
    
    # Combine normal and abnormal data
    temperature = np.concatenate([normal_temperature, abnormal_temperature])
    humidity = np.concatenate([normal_humidity, abnormal_humidity])
    sound_volume = np.concatenate([normal_sound, abnormal_sound])
    labels = np.concatenate([normal_labels, abnormal_labels])
    
    # Create DataFrame
    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'sound_volume': sound_volume,
        'anomaly': labels
    })
    
    # Shuffle the data (to mix normal and abnormal records)
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Save the data to a CSV file
    if save_to_file:
        data.to_csv(file_name, index=False)
        print(f"Dataset saved to {file_name}")
    
    return data

# 2. Simulate a continuous data stream (printing chunks to mimic real-time streaming)
def simulate_continuous_stream(data, delay=0.01):
    print("\nSimulating real-time data stream...\n")
    for i, row in data.iterrows():
        print(f"Data Point {i + 1}: Temperature={row['temperature']}, Humidity={row['humidity']}, Sound Volume={row['sound_volume']},Anomaly={row['anomaly']}")
        time.sleep(delay)  # Introduce a small delay to simulate streaming
    
        # Stop after a few data points to demonstrate (remove this for real continuous streaming)
        if i >= 20:  # Limit for demo purposes
            break

# 3. Main function to generate the data and simulate streaming
def main():
    # Generate the dataset with 5000 normal and 1600 abnormal data points
    data = generate_sensor_data(n_normal=5000, n_abnormal=1600, save_to_file=True, file_name='sensor_data.csv')
    
    # Simulate continuous stream of the generated data
    simulate_continuous_stream(data)

# Run the main function
if __name__ == "__main__":
    main() 

