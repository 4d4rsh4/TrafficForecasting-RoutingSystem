"""
Train GCN-GRU with TIME as a proper input feature
Time (hour, day) is concatenated with traffic flow for each timestep
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("="*60)
print("SPATIO-TEMPORAL GCN-GRU WITH TIME FEATURES")
print("="*60)

# --- LOAD DATA ---
print("\n1. Loading PeMS dataset...")
raw_data = np.load('data/pems04.npz')['data']
print(f"   Raw shape: {raw_data.shape}")  # (T, N, F) = (time, sensors, features)

# Get traffic flow data
data = raw_data[:, :, 0]  # (T, N) = (17856, 307)
T, N = data.shape
print(f"   Traffic flow shape: {data.shape}")

# Create time index
time_index = pd.date_range("2018-01-01", periods=T, freq='5min')
print(f"   Time range: {time_index[0]} to {time_index[-1]}")

# --- SCALE TRAFFIC DATA ---
print("\n2. Normalizing traffic flow...")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

# --- EXTRACT TIME FEATURES (CYCLICAL ENCODING) ---
print("3. Extracting time features...")
# Use sin/cos encoding for cyclical features (Monday→Sunday is circular, 23:00→00:00 is circular)
hours_rad = 2 * np.pi * time_index.hour.values / 24.0
days_rad = 2 * np.pi * time_index.dayofweek.values / 7.0

hours_sin = np.sin(hours_rad).astype(np.float32)
hours_cos = np.cos(hours_rad).astype(np.float32)
days_sin = np.sin(days_rad).astype(np.float32)
days_cos = np.cos(days_rad).astype(np.float32)

print(f"   Hour encoding: sin/cos shapes {hours_sin.shape}, {hours_cos.shape}")
print(f"   Day encoding: sin/cos shapes {days_sin.shape}, {days_cos.shape}")

# --- BUILD SEQUENCES ---
print("\n4. Building sequences...")
seq_in = 12  # Input sequence length (60 minutes)
seq_out = 12  # Output forecast horizon (60 minutes)

X = []  # Shape: (num_samples, seq_in, N, 3) - [traffic, hour, day]
Y = []  # Shape: (num_samples, seq_out, N)
T_in = []  # Time features for input

for t in range(T - seq_in - seq_out):
    # Input: traffic + time features for seq_in timesteps
    x_traffic = data_scaled[t:t+seq_in]  # (seq_in, N)
    x_hour = hours_sin[t:t+seq_in]  # (seq_in,)
    x_day = days_sin[t:t+seq_in]  # (seq_in,)
    
    # Stack: each sensor gets traffic value + hour_sin + hour_cos + day_sin + day_cos
    x_combined = np.stack([x_traffic.T, 
                           np.tile(x_hour, (N, 1)),
                           np.tile(x_day, (N, 1))], axis=-1)  # (N, seq_in, 3)
    
    # Output: traffic values for seq_out timesteps ahead
    y = data_scaled[t+seq_in:t+seq_in+seq_out]  # (seq_out, N)
    
    X.append(x_combined)
    Y.append(y)
    T_in.append([hours_sin[t+seq_in-1], days_sin[t+seq_in-1]])  # Last time in input window

X = np.array(X)  # (num_samples, N, seq_in, 3)
Y = np.array(Y)  # (num_samples, seq_out, N)
T_in = np.array(T_in)  # (num_samples, 2)

print(f"   X shape: {X.shape} (samples, sensors, seq, features)")
print(f"   Y shape: {Y.shape} (samples, seq_out, sensors)")
print(f"   T_in shape: {T_in.shape}")

# Reshape for model: (samples, seq, sensors, 3)
X = np.transpose(X, (0, 2, 1, 3))  # (num_samples, seq_in, N, 3)
print(f"   X reshaped: {X.shape}")

# --- ADJACENCY MATRIX ---
print("\n5. Loading adjacency matrix...")
try:
    dist = pd.read_csv('data/distance04.csv', header=None).values
    adj = torch.FloatTensor((dist > 0).astype(float))
    print(f"   Adjacency shape: {adj.shape}")
except Exception as e:
    print(f"   Warning: {e}, using identity")
    adj = torch.eye(N)

# --- MODEL ---
class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feat, out_feat))
        self.bias = nn.Parameter(torch.zeros(out_feat))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        # x: (B, N, in_feat)
        # adj: (N, N)
        x = torch.mm(x.view(-1, x.size(-1)), self.weight) + self.bias
        x = x.view(-1, adj.size(0), self.weight.size(-1))
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return x

class TemporalGCNGRU(nn.Module):
    def __init__(self, num_sensors, gcn_hid, gru_hid, seq_in, seq_out, adj):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_out = seq_out
        self.adj = adj
        
        # Input has 3 features (traffic + hour + day)
        self.gcn1 = GraphConvLayer(3, gcn_hid)
        self.gcn2 = GraphConvLayer(gcn_hid, gcn_hid)
        
        # GRU on spatial-temporal features
        self.gru = nn.GRU(gcn_hid, gru_hid, num_layers=1, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(gru_hid, self.seq_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (B, seq_in, N, 3)
        B, L, N, F = x.shape
        
        # Process each timestep through GCN
        gcn_out = []
        for t in range(L):
            xt = x[:, t, :, :]  # (B, N, 3)
            ht = self.gcn1(xt, self.adj)
            ht = self.relu(ht)
            ht = self.gcn2(ht, self.adj)  # (B, N, gcn_hid)
            gcn_out.append(ht)
        
        # Stack: (B, L, N, gcn_hid)
        gcn_seq = torch.stack(gcn_out, dim=1)
        
        # Keep per-sensor features (don't average)
        # gcn_seq: (B, L, N, gcn_hid)
        
        # GRU on each sensor independently
        B, L, N, H = gcn_seq.shape
        gcn_seq_reshaped = gcn_seq.view(B * N, L, H)  # (B*N, L, H)
        
        _, h = self.gru(gcn_seq_reshaped)  # h: (1, B*N, gru_hid)
        
        # Predict for each sensor
        out = self.fc(h.squeeze(0))  # (B*N, self.seq_out)
        
        # Reshape back to (B, self.seq_out, N)
        out = out.view(B, N, self.seq_out).permute(0, 2, 1)  # (B, self.seq_out, N)
        
        return out

# --- TRAINING ---
print("\n6. Training setup...")
device = torch.device('cpu')
batch_size = 32
learning_rate = 0.001
num_epochs = 30

model = TemporalGCNGRU(N, gcn_hid=32, gru_hid=64, 
                       seq_in=seq_in, seq_out=seq_out, adj=adj.to(device))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print(f"   Model params: {sum(p.numel() for p in model.parameters())}")
print(f"   Device: {device}")

# Split data
n_train = int(0.7 * len(X))
n_val = int(0.85 * len(X))

X_train = torch.FloatTensor(X[:n_train])
Y_train = torch.FloatTensor(Y[:n_train])

X_val = torch.FloatTensor(X[n_train:n_val])
Y_val = torch.FloatTensor(Y[n_train:n_val])

X_test = torch.FloatTensor(X[n_val:])
Y_test = torch.FloatTensor(Y[n_val:])

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Training loop
print("\n7. Training...\n")
best_val_loss = float('inf')
patience = 5
patience_cnt = 0

import time

for epoch in range(num_epochs):
    epoch_start = time.time()

    # Train
    model.train()
    train_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size].to(device)
        batch_y = Y_train[i:i+batch_size].to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= max(1, len(X_train) // batch_size)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch_x = X_val[i:i+batch_size].to(device)
            batch_y = Y_val[i:i+batch_size].to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item()

    val_loss /= max(1, len(X_val) // batch_size)
    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch+1:2d}/{num_epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | time: {epoch_time:.1f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_cnt = 0
        torch.save(model.state_dict(), 'traffic_model_temporal.pth')
        print("            ✓ Model saved")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Test
print("\n8. Testing...")
model.load_state_dict(torch.load('traffic_model_temporal.pth'))
model.eval()
test_loss = 0
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size].to(device)
        batch_y = Y_test[i:i+batch_size].to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        test_loss += loss.item()

test_loss /= max(1, len(X_test) // batch_size)
print(f"   Test Loss: {test_loss:.5f}")

print("\n✓ Model saved: traffic_model_temporal.pth")
print("="*60)
