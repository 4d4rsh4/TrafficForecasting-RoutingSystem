import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ---
SEQ_IN, SEQ_OUT = 12, 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. MODEL DEFINITION (Must match app.py exactly to load weights) ---
class GraphConv(nn.Module):
    def __init__(self, in_f, out_f):
        super(GraphConv, self).__init__()
        self.w = nn.Parameter(torch.randn(in_f, out_f))
        nn.init.xavier_uniform_(self.w)
    def forward(self, x, adj):
        return torch.matmul(adj, torch.matmul(x, self.w))

class TemporalGCNGRU(nn.Module):
    def __init__(self, in_f, g_hid, r_hid, out_len, adj):
        super(TemporalGCNGRU, self).__init__()
        self.register_buffer('adj_matrix', adj)
        self.gcn = GraphConv(in_f, g_hid)
        self.gru = nn.GRU(g_hid, r_hid, batch_first=True)
        self.fc = nn.Linear(r_hid, out_len)
        self.relu = nn.ReLU()
    def forward(self, x):
        B, L, N, F = x.shape
        x = x.reshape(B * L, N, F) 
        x = self.relu(self.gcn(x, self.adj_matrix))
        x = x.view(B, L, N, -1).transpose(1, 2).reshape(B * N, L, -1)
        _, h = self.gru(x)
        out = self.fc(h.squeeze(0))
        return out.view(B, N, -1).transpose(1, 2)

# --- 3. DATA LOADING & PREPROCESSING ---
print("Loading PeMS-04 dataset...")
raw = np.load('data/pems04.npz')['data']
traffic = raw[:,:, 0]
T, N = traffic.shape

# Setup Time Context
time_index = pd.date_range("2018-01-01", periods=T, freq='5min')
hours_sin = np.sin(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
days_sin = np.sin(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)

# Normalize
scaler = StandardScaler()
traffic_scaled = scaler.fit_transform(traffic.reshape(-1, 1)).reshape(traffic.shape)

# Create Adjacency Matrix
adj = torch.eye(N) # Dummy identity for shape check, will load real weights

# --- 4. LOAD THE TRAINED MODEL ---
print(f"Loading weights from 'traffic_model_temporal.pth' on {DEVICE}...")
model = TemporalGCNGRU(in_f=3, g_hid=32, r_hid=64, out_len=SEQ_OUT, adj=adj).to(DEVICE)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('traffic_model_temporal.pth'), strict=False)
else:
    model.load_state_dict(torch.load('traffic_model_temporal.pth', map_location='cpu'), strict=False)
model.eval()

# --- 5. GENERATE PREDICTIONS FOR A 24-HOUR PERIOD (288 time steps) ---
print("Generating 24-hour forecast...")
test_start = int(T * 0.8) # Start at the beginning of the test set
num_steps_to_plot = 288 # Exactly 24 hours

actual_flow = []
predicted_flow = []

with torch.no_grad():
    for i in range(test_start, test_start + num_steps_to_plot):
        # Prepare input window (the 12 steps leading up to 'i')
        input_list = []
        for t in range(i - SEQ_IN, i):
            feat = np.stack([traffic_scaled[t], np.full(N, hours_sin[t]), np.full(N, days_sin[t])], axis=1)
            input_list.append(feat)
        
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        
        # Predict
        pred_scaled = model(input_tensor)
        
        # Inverse transform only the first prediction step (t+1) for Sensor 0
        pred_real = scaler.inverse_transform(pred_scaled[0, 0, :].cpu().numpy().reshape(-1, 1)).flatten()
        actual_real = traffic[i]
        
        # Store Sensor 0 values
        predicted_flow.append(pred_real[0])
        actual_flow.append(actual_real[0])

# --- 6. VISUALIZATION ---
print("Creating visualization...")
plt.figure(figsize=(15, 6))
plt.plot(actual_flow, label="Actual Flow (Sensor 0)", color='#1f77b4', linewidth=1.5, alpha=0.7)
plt.plot(predicted_flow, label="AI Predicted Flow", color='#ef4444', linestyle='--', linewidth=2)

plt.title("24-Hour Traffic Flow Prediction Output (GCN-GRU Model)", fontsize=14, fontweight='bold')
plt.xlabel("Time (5-minute intervals)", fontsize=12)
plt.ylabel("Number of Vehicles", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Save for the report
plt.savefig('outputs/ai_prediction_output.png', dpi=300, bbox_inches='tight')
print("✅ Success! Output graph saved as 'ai_prediction_output.png'.")
plt.show()
