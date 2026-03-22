import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import folium
import os
import traceback

# --- 1. GLOBAL SETTINGS ---
SEQ_IN = 12
SEQ_OUT = 12
DEVICE = torch.device('cpu')

app = Flask(__name__)
CORS(app)

# --- 2. LOAD DATA & BUILD ADJACENCY ---
try:
    raw = np.load('data/pems04.npz')['data']
    traffic = raw[:, :, 0]  # (Timesteps, Sensors)
    T, N = traffic.shape
    
    dist_df = pd.read_csv('data/distance04.csv') 
    adj = torch.zeros((N, N))
    for _, row in dist_df.iterrows():
        u, v = int(row.iloc[0]), int(row.iloc[1])
        if u < N and v < N:
            adj[u, v] = 1.0
    adj = adj + torch.eye(N)
    
    time_index = pd.date_range("2018-01-01", periods=T, freq='5min')
    hours_sin = np.sin(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
    days_sin = np.sin(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)
    
    scaler = StandardScaler()
    traffic_scaled = scaler.fit_transform(traffic.reshape(-1, 1)).reshape(traffic.shape)
    print(f"✅ Data Loaded: {N} sensors.")
except Exception as e:
    print(f"❌ Data Error: {e}")
    T, N = 1000, 307
    adj = torch.eye(N)

# --- 3. MODEL ARCHITECTURE ---
class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feat, out_feat))
        self.bias = nn.Parameter(torch.zeros(out_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight) + self.bias
        return out

class TemporalGCNGRU(nn.Module):
    def __init__(self, num_sensors, gcn_hid, gru_hid, seq_in, seq_out, adj):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_out = seq_out
        self.adj = adj
        # REVERTED TO 3 FEATURES (Flow, Hour, Day) TO MATCH YOUR .PTH FILE
        self.gcn1 = GraphConvLayer(3, gcn_hid) 
        self.gcn2 = GraphConvLayer(gcn_hid, gcn_hid)
        self.gru = nn.GRU(gcn_hid, gru_hid, num_layers=1, batch_first=True)
        self.fc = nn.Linear(gru_hid, self.seq_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, L, N, F = x.shape
        gcn_out = []
        for t in range(L):
            xt = x[:, t, :, :]
            ht = self.relu(self.gcn1(xt, self.adj))
            ht = self.gcn2(ht, self.adj)
            gcn_out.append(ht)
        gcn_seq = torch.stack(gcn_out, dim=1)
        B, L, N, H = gcn_seq.shape
        gcn_seq_reshaped = gcn_seq.view(B * N, L, H)
        _, h = self.gru(gcn_seq_reshaped)
        out = self.fc(h.squeeze(0)) 
        out = out.view(B, N, self.seq_out).permute(0, 2, 1) 
        return out

# --- 4. INSTANTIATE & LOAD MODEL ---
model = TemporalGCNGRU(num_sensors=N, gcn_hid=32, gru_hid=64, seq_in=SEQ_IN, seq_out=SEQ_OUT, adj=adj.to(DEVICE))
model_path = 'traffic_model_temporal.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ Advanced Temporal GCN-GRU Model Loaded.")
model.eval()

# --- 5. CACHE SAN FRANCISCO MAP ---
print("Loading San Francisco Graph...")
if os.path.exists("sf_map.graphml"):
    GLOBAL_GRAPH = ox.load_graphml("sf_map.graphml")
else:
    print("Downloading Map... (Run setup_map.py next time to make this instant)")
    GLOBAL_GRAPH = ox.graph_from_address("San Francisco, California", dist=15000, network_type='drive')
print("✅ Graph loaded successfully! Server is ready.")


# --- 6. OVERVIEW MAP ENDPOINT (FOR HOMEPAGE) ---
@app.route('/get_overview', methods=['POST'])
def get_overview():
    try:
        payload = request.json
        target_day_str = payload.get('day', 'Friday')
        target_hour = int(payload.get('hour', 17))
        
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        target_day_int = day_map.get(target_day_str, 4)
        
        search_start = int(T * 0.7)
        found_idx = search_start
        for i in range(search_start, T - SEQ_IN):
            if time_index[i].dayofweek == target_day_int and time_index[i].hour == target_hour:
                found_idx = i
                break

        # REVERTED TO 3 FEATURES
        input_list = []
        for t in range(found_idx, found_idx + SEQ_IN):
            feat = np.stack([
                traffic_scaled[t], 
                np.full(N, hours_sin[t]), 
                np.full(N, days_sin[t])    
            ], axis=1)
            input_list.append(feat)
            
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0, :].cpu().numpy().reshape(-1, 1)).flatten()

        m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles='CartoDB dark_matter')
        G = GLOBAL_GRAPH
        
        # Draw heatmap on major highways
        for u, v, k, data in G.edges(keys=True, data=True):
            highway_type = data.get('highway')
            if isinstance(highway_type, list): highway_type = highway_type[0]
            
            if highway_type in ['motorway', 'motorway_link', 'trunk']:
                sensor_id = (u + v) % N
                flow = predicted_flows[sensor_id]
                
                color = '#10b981' if flow < 200 else '#f59e0b' if flow < 400 else '#ef4444'
                
                if 'geometry' in data:
                    coords = [[lat, lon] for lon, lat in data['geometry'].coords]
                else:
                    coords = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
                
                folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)

        # Overview Legend
        overview_legend = '''
        <div style="position: absolute; top: 90px; left: 20px; width: 140px; background-color: rgba(9, 9, 11, 0.9); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; z-index: 999999; font-size: 12px; color: white; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); font-family: 'Inter', sans-serif;">
            <b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:8px; letter-spacing: 0.5px;">Traffic Flow</b>
            <div style="margin-bottom:6px; display:flex; align-items:center;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Free</div>
            <div style="margin-bottom:6px; display:flex; align-items:center;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Moderate</div>
            <div style="display:flex; align-items:center;"><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Heavy</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(overview_legend))
        return jsonify({'map_html': m._repr_html_()})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 7. SIMULATION ROUTING ENDPOINT ---
@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        payload = request.json
        start_lat, start_lon = float(payload['start_lat']), float(payload['start_lon'])
        end_lat, end_lon = float(payload['end_lat']), float(payload['end_lon'])

        target_day_str = payload.get('day', 'Friday')
        target_hour = int(payload.get('hour', 17))
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        target_day_int = day_map.get(target_day_str, 4)
        
        search_start = int(T * 0.7)
        found_idx = search_start
        for i in range(search_start, T - SEQ_IN):
            if time_index[i].dayofweek == target_day_int and time_index[i].hour == target_hour:
                found_idx = i
                break

        # REVERTED TO 3 FEATURES
        input_list = []
        for t in range(found_idx, found_idx + SEQ_IN):
            feat = np.stack([
                traffic_scaled[t], 
                np.full(N, hours_sin[t]), 
                np.full(N, days_sin[t])    
            ], axis=1)
            input_list.append(feat)
            
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0, :].cpu().numpy().reshape(-1, 1)).flatten()

        # Grab a subgraph specific to the routing area from our CACHED map
        dist_buffer = 0.04
        bbox = (min(start_lon, end_lon) - dist_buffer, min(start_lat, end_lat) - dist_buffer, 
                max(start_lon, end_lon) + dist_buffer, max(start_lat, end_lat) + dist_buffer)
        G = ox.truncate.truncate_graph_bbox(GLOBAL_GRAPH, bbox=bbox)
        
        for u, v, k, data in G.edges(keys=True, data=True):
            sensor_id = (u + v) % N
            flow = predicted_flows[sensor_id]
            highway_type = data.get('highway')
            if isinstance(highway_type, list): highway_type = highway_type[0]
            
            is_highway = highway_type in ['motorway', 'trunk', 'primary', 'motorway_link']
            if is_highway:
                data['flow_val'] = flow * 1.2
                capacity, speed = 1200.0, 65.0 
            else:
                data['flow_val'] = flow * 0.4
                capacity, speed = 400.0, 35.0  

            congestion_ratio = data['flow_val'] / capacity
            base_time = data['length'] / (speed / 3.6)
            physical_time = base_time * (1 + 1.5 * (congestion_ratio)**4)

            # Routing Penalty Logic
            if congestion_ratio >= 0.7: comfort_penalty = 5.0 
            elif congestion_ratio >= 0.4: comfort_penalty = 2.0
            else: comfort_penalty = 1.0

            data['travel_time'] = physical_time * comfort_penalty
            data['distance_weight'] = data['length']

        n_start = ox.distance.nearest_nodes(G, start_lon, start_lat)
        n_end = ox.distance.nearest_nodes(G, end_lon, end_lat)
        route_shortest = nx.shortest_path(G, n_start, n_end, weight='distance_weight')
        route_ai = nx.shortest_path(G, n_start, n_end, weight='travel_time')

        def get_route_stats(graph, route):
            dists, times = [], []
            for u, v in zip(route[:-1], route[1:]):
                edge_data = graph.get_edge_data(u, v)[0]
                dists.append(edge_data.get('length', 0))
                times.append(edge_data.get('travel_time', 0)) 
            return sum(dists), sum(times)

        s_dist_m, s_time_s = get_route_stats(G, route_shortest)
        a_dist_m, a_time_s = get_route_stats(G, route_ai)
        time_saved_mins = max(0, (s_time_s - a_time_s) / 60.0)

        # Build Map
        m = folium.Map(tiles='CartoDB dark_matter')
        
        # Heatmap
        for u, v, k, data in G.edges(keys=True, data=True):
            f = data['flow_val']
            color = '#22c55e' if f < 250 else '#f59e0b' if f < 550 else '#ef4444'
            if 'geometry' in data:
                coords = [[lat, lon] for lon, lat in data['geometry'].coords]
            else:
                coords = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
            folium.PolyLine(coords, color=color, weight=2, opacity=0.3).add_to(m)

        # Routes
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in route_ai], color='#06b6d4', weight=8, opacity=1.0).add_to(m)
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in route_shortest], color='#FFFFFF', weight=3, opacity=1.0, dash_array='10, 15').add_to(m)

        folium.Marker([start_lat, start_lon], icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)

        # Auto-frame map
        m.fit_bounds([[start_lat, start_lon], [end_lat, end_lon]])

        # Legend / Stats Box (Top Right)
        stats_html = f'''
             <div style="position: absolute; top: 20px; right: 20px; width: 240px; 
                         background-color: rgba(15, 23, 42, 0.95); color: white; 
                         border: 2px solid #06b6d4; z-index: 9999; padding: 15px; 
                         border-radius: 10px; font-family: 'Inter', sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);">
                 <h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Simulation Results</h4>
                 <div style="font-size: 12px; line-height: 1.6;">
                     <b>AI Optimized Route:</b> {a_time_s/60:.1f} mins | {a_dist_m/1000:.1f} km<br>
                     <b>Shortest Route:</b> {s_time_s/60:.1f} mins | {s_dist_m/1000:.1f} km<br>
                     <b style="color: #22c55e;">Time Saved: {time_saved_mins:.1f} mins</b>
                 </div>
                 
                 <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;">
                 
                 <b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Predicted Traffic</b>
                 <div style="margin-bottom:4px; display:flex; align-items:center;"><i style="background:#22c55e; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Free Flow</div>
                 <div style="margin-bottom:4px; display:flex; align-items:center;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Moderate</div>
                 <div style="display:flex; align-items:center;"><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Heavy Congestion</div>
             </div>
        '''
        m.get_root().html.add_child(folium.Element(stats_html))
        return jsonify({'map_html': m._repr_html_()})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)