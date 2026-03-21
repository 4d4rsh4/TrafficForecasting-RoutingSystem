import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# STAGE 1: DATA LOADING & FEATURE ENGINEERING
# ==========================================
def prepare_data(npz_path, dist_path, num_nodes=307):
    # Load Traffic Flow
    raw_data = np.load(npz_path)['data'][:, :num_nodes, 0] # (Time, Nodes)
    
    # Create Time Features (Hour of day, Day of week)
    # 16992 steps of 5-min intervals
    time_index = pd.date_range("2018-01-01", periods=raw_data.shape[0], freq='5min')
    hours = (time_index.hour.values / 23.0).reshape(-1, 1)
    days = (time_index.dayofweek.values / 6.0).reshape(-1, 1)
    
    # Scale Traffic Data
    scaler = StandardScaler()
    flow_scaled = scaler.fit_transform(raw_data)
    
    # Combine [Flow (307) + Hour (1) + Day (1)] = 309 inputs per timestep
    full_data = np.hstack([flow_scaled, hours, days])
    
    # Load Distance Graph
    dist_df = pd.read_csv(dist_path)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    # Gaussian Kernel for Adjacency Matrix
    std = dist_df.iloc[:, 2].std()
    for _, row in dist_df.iterrows():
        u, v, d = int(row['from']), int(row['to']), float(row['cost'])
        if u < num_nodes and v < num_nodes:
            adj[u, v] = np.exp(-d**2 / (std**2))
    
    return full_data, adj, scaler, time_index

# Sliding Window: Look at 12 steps (1hr), predict 12 steps (1hr)
def create_window(data, in_steps=12, out_steps=12, num_nodes=307):
    X, y = [], []
    for i in range(len(data) - in_steps - out_steps):
        X.append(data[i : i + in_steps, :])
        y.append(data[i + in_steps : i + in_steps + out_steps, :num_nodes])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# ==========================================
# STAGE 2: THE T-GCN MODEL (Spatial + Temporal)
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x, adj):
        # x: (Batch, Nodes, Features) | adj: (Nodes, Nodes)
        # Aggregates neighbor info: A * X * W
        out = torch.matmul(adj, x)
        return torch.matmul(out, self.weights)

class TGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_steps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Spatial Gate: Learns which neighbors matter
        self.gcn_gates = GCNLayer(input_dim + hidden_dim, hidden_dim * 2)
        # Spatial Candidate: Learns the actual traffic transformation
        self.gcn_cand = GCNLayer(input_dim + hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_steps)

    def forward(self, x_seq, adj):
        # x_seq: (Batch, Time, Nodes + TimeFeatures)
        batch_size, seq_len, _ = x_seq.shape
        hidden = torch.zeros(batch_size, self.num_nodes, self.hidden_dim).to(x_seq.device)
        
        for t in range(seq_len):
            # 1. Split traffic data and time features
            # x_t: (Batch, Nodes, 1)
            x_t = x_seq[:, t, :self.num_nodes].unsqueeze(-1)
            # 2. Add time features to every node (simple broadcast)
            # time_feat: (Batch, Nodes, 2)
            time_feat = x_seq[:, t, self.num_nodes:].unsqueeze(1).repeat(1, self.num_nodes, 1)
            
            # Combine Traffic + Time + Hidden
            combined = torch.cat([x_t, time_feat, hidden], dim=-1) # (B, N, 1+2+H)
            
            # GRU-style Update/Reset Gates using GCN
            gates = torch.sigmoid(self.gcn_gates(combined, adj))
            z, r = torch.split(gates, self.hidden_dim, dim=-1)
            
            # Candidate memory
            cand_input = torch.cat([x_t, time_feat, r * hidden], dim=-1)
            candidate = torch.tanh(self.gcn_cand(cand_input, adj))
            
            hidden = z * hidden + (1 - z) * candidate
            
        # Map hidden state to 12 future steps
        out = self.fc(hidden) # (Batch, Nodes, 12)
        return out.transpose(1, 2) # (Batch, 12, Nodes)

# ==========================================
# STAGE 3: ROUTING ALGORITHM
# ==========================================
def find_predictive_route(start_node, end_node, adj_matrix, predictions):
    """
    Dijkstra's Algorithm where:
    Cost = Predicted_Traffic + Small_Distance_Constant
    """
    G = nx.DiGraph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i, j] > 0:
                # The 'cost' of a road is the AI's predicted traffic flow at node j
                cost = float(predictions[j]) + 1.0
                G.add_edge(i, j, weight=cost)
    
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        return path
    except:
        return None

# ==========================================
# STAGE 4: MAIN PIPELINE (Execution)
# ==========================================
# 1. Prepare Data
print("Loading data and building graph...")
full_data, adj, scaler, time_index = prepare_data('data/pems04.npz', 'data/distance04.csv')
adj_tensor = torch.FloatTensor(adj)

X, y = create_window(full_data)
split = int(0.8 * len(X))
train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=32, shuffle=True)

# 2. Initialize Model
# input_dim = 1 (flow) + 2 (hour/day) = 3
model = TGCN(num_nodes=307, input_dim=3, hidden_dim=32, output_steps=12)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Training Loop
print("Starting Training (T-GCN)...")
model.train()
for epoch in range(5): # Set to 20+ for real project
    total_loss = 0
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        pred = model(b_x, adj_tensor)
        loss = criterion(pred, b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# 4. Routing Example (The Goal)
print("\n--- Routing Inference ---")
model.eval()
with torch.no_grad():
    # Pick a random sample from the test set
    sample_input = X[split:split+1] 
    # AI predicts the next 12 steps (1 hour)
    future_preds = model(sample_input, adj_tensor).squeeze(0) # (12, 307)
    
    # Let's route based on traffic predicted 30 mins from now (index 5)
    traffic_30min_ahead = future_preds[5].numpy()
    
    # Example: Find best path from Sensor 0 to Sensor 50
    start, end = 0, 50
    best_path = find_predictive_route(start, end, adj, traffic_30min_ahead)
    
    print(f"Route from {start} to {end} for next 30 mins:")
    print(f"AI Recommended Path: {best_path}")