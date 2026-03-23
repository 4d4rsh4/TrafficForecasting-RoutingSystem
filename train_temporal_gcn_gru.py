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
data = raw_data[:,:, 0]
T, N = data.shape

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

time_index = pd.date_range("2018-01-01", periods=T, freq='5min')


def get_cyclical_coords(values, max_val):
    rad = 2 * np.pi * values / max_val
    return np.sin(rad), np.cos(rad)


h_sin, h_cos = get_cyclical_coords(time_index.hour.values + time_index.minute.values / 60, 24)
d_sin, d_cos = get_cyclical_coords(time_index.dayofweek.values, 7)

seq_in, seq_out = 12, 12
X, Y = [], []

for t in range(T - seq_in - seq_out):
    xt = data_scaled[t:t + seq_in,:, np.newaxis]
    
    ts = np.stack([h_sin[t:t + seq_in], h_cos[t:t + seq_in],
                   d_sin[t:t + seq_in], d_cos[t:t + seq_in]], axis=-1)
    ts = np.tile(ts[:, np.newaxis,:], (1, N, 1)) 
    
    X.append(np.concatenate([xt, ts], axis=-1))
    Y.append(data_scaled[t + seq_in:t + seq_in + seq_out])

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
model = TemporalGCNGRU(in_f=5, g_hid=32, r_hid=64, out_len=seq_out, adj=adj_norm.to(device)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

idx = int(len(X) * 0.8)
X_train, Y_train = torch.tensor(X[:idx]), torch.tensor(Y[:idx])
X_test, Y_test = torch.tensor(X[idx:]), torch.tensor(Y[idx:])

print(f"Training on {device}...")
for epoch in range(20):
    epoch_start = time.time()
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0
    for i in range(0, len(X_train), 32):
        indices = perm[i:i + 32]
        bx, by = X_train[indices].to(device), Y_train[indices].to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/(len(X_train)//32):.5f} | Time: {epoch_time:.2f}s")