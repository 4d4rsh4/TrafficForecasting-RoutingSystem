import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

print("="*60)
print("OPTIMIZED SPATIO-TEMPORAL GCN-GRU")
print("="*60)

print("\n1. Loading and Preprocessing...")
raw_data = np.load('data/pems04.npz')['data']
data = raw_data[:,:, 0]  # (T, N)
T, N = data.shape

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # (T, N)

time_index = pd.date_range("2018-01-01", periods=T, freq='5min')
hours_sin = np.sin(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
days_sin = np.sin(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)

seq_in, seq_out = 12, 12
X, Y = [], []

for t in range(T - seq_in - seq_out):
    xt = data_scaled[t:t + seq_in,:, np.newaxis]
    
    h_feat = np.tile(hours_sin[t:t + seq_in, np.newaxis, np.newaxis], (1, N, 1))
    d_feat = np.tile(days_sin[t:t + seq_in, np.newaxis, np.newaxis], (1, N, 1))
    
    X.append(np.concatenate([xt, h_feat, d_feat], axis=-1))  # (seq_in, N, 3)
    Y.append(data_scaled[t + seq_in:t + seq_in + seq_out])  # (seq_out, N)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0))
    d = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


try:
    dist = pd.read_csv('data/distance04.csv', header=None).values
    adj = torch.FloatTensor((dist > 0).astype(float))
except:
    adj = torch.eye(N)
adj_norm = normalize_adj(adj)


class GraphConv(nn.Module):

    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_f, out_f))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, adj):
        x = torch.matmul(x, self.w) 
        return torch.matmul(adj, x)


class TemporalGCNGRU(nn.Module):

    def __init__(self, in_f, g_hid, r_hid, out_len, adj):
        super().__init__()
        self.adj = adj
        self.gcn = GraphConv(in_f, g_hid)
        self.gru = nn.GRU(g_hid, r_hid, batch_first=True)
        self.fc = nn.Linear(r_hid, out_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, L, N, F = x.shape
        x = x.reshape(B * L, N, F) 
        x = self.relu(self.gcn(x, self.adj))
        x = x.view(B, L, N, -1).transpose(1, 2).reshape(B * N, L, -1)
        _, h = self.gru(x) 
        out = self.fc(h.squeeze(0)) 
        return out.view(B, N, -1).transpose(1, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizing Hardware: {device}")

model = TemporalGCNGRU(in_f=3, g_hid=32, r_hid=64, out_len=seq_out, adj=adj_norm.to(device)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

idx = int(len(X) * 0.8)
X_train, Y_train = torch.tensor(X[:idx]), torch.tensor(Y[:idx])
X_test, Y_test = torch.tensor(X[idx:]), torch.tensor(Y[idx:])

train_loss_history = []
val_loss_history = []

num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    model.train()
    perm = torch.randperm(len(X_train))
    total_train_loss = 0
    
    for i in range(0, len(X_train), batch_size):
        indices = perm[i:i + batch_size]
        
        bx, by = X_train[indices].to(device), Y_train[indices].to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / (len(X_train) // batch_size)
    train_loss_history.append(round(avg_train_loss, 5))
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for j in range(0, len(X_test), batch_size):
            bx_val = X_test[j:j + batch_size].to(device)
            by_val = Y_test[j:j + batch_size].to(device)
            v_loss = criterion(model(bx_val), by_val)
            total_val_loss += v_loss.item()
            
    avg_val_loss = total_val_loss / (len(X_test) // batch_size)
    val_loss_history.append(round(avg_val_loss, 5))
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Time: {epoch_time:.2f}s")

torch.save(model.state_dict(), 'traffic_model_temporal.pth')

print("\n" + "="*60)
print("TRAINING COMPLETE. COPY THESE FOR YOUR PLOTTING SCRIPT:")
print("="*60)
print(f"actual_train_loss = {train_loss_history}")
print(f"actual_val_loss = {val_loss_history}")
