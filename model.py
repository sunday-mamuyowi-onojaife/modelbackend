import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_samples = 1000

# Generate normal sensor data (within typical operating ranges)
temperature = np.random.normal(loc=70, scale=5, size=n_samples)  # Mean temp 70°F, std dev 5
humidity = np.random.normal(loc=50, scale=10, size=n_samples)    # Mean humidity 50%, std dev 10
sound_volume = np.random.normal(loc=60, scale=8, size=n_samples) # Mean sound 60dB, std dev 8

# Adjust size to match the number of values that will be replaced
# Every 50th value for temperature, every 30th for humidity, every 40th for sound volume
temperature[::50] = np.random.normal(loc=100, scale=5, size=temperature[::50].shape[0])  # High temp anomalies
humidity[::30] = np.random.normal(loc=20, scale=5, size=humidity[::30].shape[0])         # Low humidity anomalies
sound_volume[::40] = np.random.normal(loc=90, scale=5, size=sound_volume[::40].shape[0]) # High sound anomalies

# Combine into a DataFrame
data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'sound_volume': sound_volume
})

# Label the anomalies: where temperature > 90, humidity < 30, or sound_volume > 80
data['anomaly'] = ((data['temperature'] > 90) | (data['humidity'] < 30) | (data['sound_volume'] > 80)).astype(int)

print(data.head())  # Display first few rows of the dataset

