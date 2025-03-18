import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('attentiveness_data.csv')

# Convert the 'Timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# Set the 'Timestamp' column as the index
data.set_index('Timestamp', inplace=True)

# Resample the data at 1-minute intervals and count the occurrences of 'Attentive' and 'Not Attentive'
resampled_data = data.groupby(pd.Grouper(freq='1T'))['Prediction'].value_counts().unstack().fillna(0)

# Plot the line graph
plt.figure(figsize=(12, 6))
resampled_data.plot(kind='line', marker='o')
plt.title('Attentiveness Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.grid(True)
plt.legend(title='Attentiveness')
plt.show()