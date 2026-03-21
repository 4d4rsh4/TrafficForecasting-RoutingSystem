import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset



# load data
# Features are flow, occupancy, speed
raw_data = np.load('data/PeMS/pems04.npz')['data']

# 0 = flow 1 = occupancy 2 = speed
# data shape = 16992 x 307
data = raw_data[:, :, 0] #slicing

print(f"Total time steps: {data.shape[0]}")
print(f"Number of sensors: {data.shape[1]}")



# 1. Create a timestamp for every one of the 16,992 steps
start_date = "2018-01-01 00:00:00"
time_index = pd.date_range(start_date, periods=16992, freq='5min')

# 2. Extract Hour (0-23) and Day of Week (0-6, where 0 is Monday)
hours = time_index.hour.values.reshape(-1, 1)
days = time_index.dayofweek.values.reshape(-1, 1)

# 3. Normalize them (Crucial!)
# Neural networks hate large numbers. Scale them between 0 and 1.
hours_scaled = hours / 23.0
days_scaled = days / 6.0


# normalize the data
scaler = StandardScaler()
# flatten to scale all sensors together, then reshape back
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

# We add the 2 new columns to the end
data_with_time = np.hstack([data_scaled, hours_scaled, days_scaled])

print(f"New Data Shape: {data_with_time.shape}") # (16992, 309)




print(f"Hours shape: {hours_scaled.shape}") # (16992, 1)
# sliding window 
def window(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):  #16992 - 12 - 12
        # ip
        X.append(data[i : i + input_steps, :])
        # op
        y.append(data[i + input_steps : i + input_steps + output_steps, :307])
    return np.array(X), np.array(y)

# 12 because 5 mins x 12 = 60 (1 hr)
X, y = window(data_with_time, 12, 12)

# 80/20 split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# convert to pytorch sensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)




class TrafficPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_steps, num_sensors):
        super(TrafficPredictor, self).__init__()
        self.output_steps = output_steps
        self.num_sensors = num_sensors # 307
        
        # Corrected GRU: input_dim is 309, hidden_dim is 64
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Linear layer maps from hidden_dim (64) to output (307 * 12)
        self.fc = nn.Linear(hidden_dim, num_sensors * output_steps)

    def forward(self, x):
        # x shape: (batch, 12, 309)
        _, hsn = self.gru(x) 
        # hsn[-1] is the last hidden state: (batch, 64)
        out = self.fc(hsn[-1]) 
        # Reshape to (batch, 12, 307)
        return out.view(x.shape[0], self.output_steps, self.num_sensors)
    


# Initialize
model = TrafficPredictor(input_dim=309, hidden_dim=64, num_layers=2, output_steps=12, num_sensors=307)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

for epoch in range(10): 
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        prediction = model(batch_X)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")


import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_performance(model, X_test, y_test, scaler):
    model.eval() # 1. Set model to "Testing Mode"
    
    with torch.no_grad(): # 2. Turn off gradient tracking (saves memory)
        # 3. Get predictions
        y_pred_scaled = model(X_test)
        
        # 4. Inverse Transform (Convert back to "Real World" car counts)
        # We must reshape to 2D for the scaler, then back to 3D
        y_pred_rescaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, 307)).reshape(y_pred_scaled.shape)
        y_true_rescaled = scaler.inverse_transform(y_test.reshape(-1, 307)).reshape(y_test.shape)
        
    # 5. Calculate Metrics for the first 15-min prediction (index 0)
    mae = mean_absolute_error(y_true_rescaled[:, 0, :], y_pred_rescaled[:, 0, :])
    rmse = np.sqrt(mean_squared_error(y_true_rescaled[:, 0, :], y_pred_rescaled[:, 0, :]))
    
    print(f"--- Final Test Results ---")
    print(f"MAE: {mae:.2f} cars")
    print(f"RMSE: {rmse:.2f} cars")
    
    return y_true_rescaled, y_pred_rescaled

# Run the evaluation
y_true, y_pred = evaluate_performance(model, X_test, y_test, scaler)



# visualization
import matplotlib.pyplot as plt

# for sensor 0
sensor_idx = 0
time_steps = 288 

plt.figure(figsize=(15, 6))
plt.plot(y_true[:time_steps, 0, sensor_idx], label="Actual Flow", color='blue', alpha=0.7)
plt.plot(y_pred[:time_steps, 0, sensor_idx], label="Predicted Flow", color='red', linestyle='--')

plt.title(f"24-Hour Traffic Flow Prediction (Sensor {sensor_idx})")
plt.xlabel("Time (5-min intervals)")
plt.ylabel("Number of Vehicles")
plt.legend()
plt.grid(True)
plt.show()



def get_prediction_by_user_input(input_day_name, input_hour):
    """
    input_day_name: 'Monday', 'Tuesday', etc.
    input_hour: 0-23
    """
    # Map day names to numbers (0=Monday, 6=Sunday)
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    target_day = day_map[input_day_name]

    # Find the first index in the data that matches this day and hour
    # We search in the test set portion
    search_start = split # Start looking in the test data
    found_idx = -1
    
    for i in range(search_start, len(time_index) - 12):
        if time_index[i].dayofweek == target_day and time_index[i].hour == input_hour:
            found_idx = i
            break
            
    if found_idx == -1:
        return "Sorry, that day/hour combination was not found in the test data."
    
    # Get the "Window" (the past 12 steps leading up to this hour)
    # We need the 309 columns (flow + hour_scaled + day_scaled)
    window_data = data_with_time[found_idx - 12 : found_idx, :]
    window_tensor = torch.FloatTensor(window_data).unsqueeze(0) # Add batch dimension

    # Model Predict
    model.eval()
    with torch.no_grad():
        pred_scaled = model(window_tensor)
        # Rescale only the first prediction (15 mins ahead)
        pred_rescaled = scaler.inverse_transform(pred_scaled[0].reshape(-1, 307))
        flow_prediction = pred_rescaled[0, 0] # Sensor 0


    print(f"\nPREDICTION SYSTEM")
    print(f"Day: {input_day_name} | Hour: {input_hour}:00")
    print(f"Predicted Flow for next 15 mins: {flow_prediction:.1f} vehicles")
 

#INTERACTIVE SECTION

my_day = "Friday" 
my_hour = 17 

get_prediction_by_user_input(my_day, my_hour)

