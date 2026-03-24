import time
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

SEQ_IN, SEQ_OUT = 12, 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)

# --- DATA & MODEL INITIALIZATION ---
try:
    raw = np.load('data/pems04.npz')['data']
    traffic = raw[:,:, 0]
    T, N = traffic.shape
    dist_df = pd.read_csv('data/distance04.csv') 
    adj = torch.zeros((N, N))
    for _, row in dist_df.iterrows():
        u, v = int(row.iloc[0]), int(row.iloc[1])
        if u < N and v < N: adj[u, v] = 1.0
    adj = adj + torch.eye(N)
    time_index = pd.date_range("2018-01-01", periods=T, freq='5min')
    hours_sin = np.sin(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
    days_sin = np.sin(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)
    scaler = StandardScaler()
    traffic_scaled = scaler.fit_transform(traffic.reshape(-1, 1)).reshape(traffic.shape)
    print(f"✅ Data Loaded: {N} sensors on {DEVICE}.")
except Exception as e:
    print(f"❌ Data Error: {e}")
    T, N = 1000, 307
    adj = torch.eye(N)


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


model = TemporalGCNGRU(in_f=3, g_hid=32, r_hid=64, out_len=SEQ_OUT, adj=adj).to(DEVICE)
model_path = 'traffic_model_temporal.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    print("✅ Advanced Model Loaded.")
model.eval()


# --- ENDPOINTS ---
@app.route('/get_overview', methods=['POST', 'GET'])
def get_overview():
    try:
        # Get AI prediction
        found_idx = int(T * 0.8) + 100 
        input_list = [np.stack([traffic_scaled[t], np.full(N, hours_sin[t]), np.full(N, days_sin[t])], axis=1) for t in range(found_idx, found_idx + SEQ_IN)]
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0,:].cpu().numpy().reshape(-1, 1)).flatten()
        
        # --- NEW LOGIC: DOWNLOAD ONLY MAJOR HIGHWAYS ---
        file_name = "sf_highway_map.graphml"
        if os.path.exists(file_name):
            G = ox.load_graphml(file_name)
        else:
            print("Downloading HIGHWAY-ONLY map for San Francisco...")
            # This filter tells OSMnx to ONLY get these road types
            custom_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary"]'
            G = ox.graph_from_place("San Francisco, California", custom_filter=custom_filter)
            ox.save_graphml(G, filepath=file_name)

        m = folium.Map(tiles='CartoDB dark_matter')
        
        # Paint the highways with AI predictions
        edge_index = 0
        for u, v, _, data in G.edges(keys=True, data=True):
            flow = predicted_flows[edge_index % N]
            color = '#10b981' if flow < 200 else '#f59e0b' if flow < 400 else '#ef4444'
            
            # Slightly thicker lines to match your reference image
            weight = 5
            
            coords = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            folium.PolyLine(coords, color=color, weight=weight, opacity=0.9).add_to(m)
            edge_index += 1

        # Zoom in on the specific highway network
        m.fit_bounds([[37.7, -122.5], [37.81, -122.38]]) 
        
        # Add Legends & Overlays
        legend = '''<div style="position: absolute; top: 90px; left: 20px; width: 140px; background-color: rgba(9, 9, 11, 0.9); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; z-index: 999999; font-size: 12px; color: white; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:8px;">Traffic Flow</b><div style="margin-bottom:6px;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free</div><div style="margin-bottom:6px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy</div></div>'''
        live_state = '''<div style="position: absolute; top: 20px; right: 20px; width: 250px; background-color: rgba(9, 9, 11, 0.9); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; z-index: 999999; font-size: 12px; color: #a1a1aa; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);"><b style="font-size:14px; color:white; display:block; margin-bottom:5px;">Live Traffic State</b>This map shows current traffic conditions in the San Francisco Bay Area using our AI model. Green indicates smooth traffic; red indicates congestion.</div>'''
        
        m.get_root().html.add_child(folium.Element(legend))
        m.get_root().html.add_child(folium.Element(live_state))
        
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    
@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        start_time = time.time()
        payload = request.json
        start_lat, start_lon = float(payload['start_lat']), float(payload['start_lon'])
        end_lat, end_lon = float(payload['end_lat']), float(payload['end_lon'])
        
        target_day_int = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}.get(payload.get('day', 'Friday'), 4)
        target_hour = int(payload.get('hour', 17))
        
        search_start = int(T * 0.7)
        found_idx = search_start
        for i in range(search_start, T - SEQ_IN):
            if time_index[i].dayofweek == target_day_int and time_index[i].hour == target_hour:
                found_idx = i
                break
        input_list = [np.stack([traffic_scaled[t], np.full(N, hours_sin[t]), np.full(N, days_sin[t])], axis=1) for t in range(found_idx, found_idx + SEQ_IN)]
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0,:].cpu().numpy().reshape(-1, 1)).flatten()
        
        print(f"\n--- Dynamic Map Fetch for route: ({start_lat},{start_lon}) -> ({end_lat},{end_lon}) ---")
        dist_buffer = 0.02 
        bbox = (
            min(start_lon, end_lon) - dist_buffer, min(start_lat, end_lat) - dist_buffer,
            max(start_lon, end_lon) + dist_buffer, max(start_lat, end_lat) + dist_buffer
        )
        G = ox.graph_from_bbox(bbox=bbox, network_type='drive')

        n_start = ox.distance.nearest_nodes(G, start_lon, start_lat)
        n_end = ox.distance.nearest_nodes(G, end_lon, end_lat)

        avg_flow = np.mean(predicted_flows)
        trend = max(0.5, avg_flow / 200.0) 

        # We no longer use a 'city_id' check here, applying the same logic to all dynamic maps
        for u, v, _, data in G.edges(keys=True, data=True):
            hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            is_h = hw in ['motorway', 'trunk', 'primary', 'motorway_link']
            
            # Apply Heuristic Topology Transfer based on model's time trend
            if is_h: base_flow = 800
            elif hw in ['secondary', 'tertiary']: base_flow = 300
            else: base_flow = 100
            geo_noise = 0.8 + 0.4 * ((u % 10) / 10.0)
            f = base_flow * trend * geo_noise
            
            data['flow_val'] = f * 1.2 if is_h else f * 0.4
            cap, spd = (1200.0, 65.0) if is_h else (400.0, 35.0)
            ratio = data['flow_val'] / cap
            tm = (data.get('length', 100) / (spd / 3.6)) * (1 + 1.5 * (ratio) ** 4)
            pen = 5.0 if ratio >= 0.7 else 2.0 if ratio >= 0.4 else 1.0
            data['travel_time'] = tm * pen
            data['distance_weight'] = data.get('length', 100)

        try:
            r_s = nx.shortest_path(G, n_start, n_end, weight='distance_weight')
            r_a = nx.shortest_path(G, n_start, n_end, weight='travel_time')
        except nx.NetworkXNoPath:
            return jsonify({'error': "No valid path found. Try moving markers slightly."}), 400

        def get_stats(g, r):
            dists, times = zip(*[(g.get_edge_data(u, v)[0].get('length', 0), g.get_edge_data(u, v)[0].get('travel_time', 0)) for u, v in zip(r[:-1], r[1:])])
            return sum(dists), sum(times)

        s_d, s_t = get_stats(G, r_s)
        a_d, a_t = get_stats(G, r_a)

        m = folium.Map(tiles='CartoDB dark_matter')
        for u, v, _, data in G.edges(keys=True, data=True):
            val = data['flow_val']
            clr = '#22c55e' if val < 250 else '#f59e0b' if val < 550 else '#ef4444'
            c = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            folium.PolyLine(c, color=clr, weight=2, opacity=0.3).add_to(m)

        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in r_a], color='#06b6d4', weight=6, opacity=1.0).add_to(m)
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in r_s], color='#FFFFFF', weight=3, opacity=0.7, dash_array='10, 15').add_to(m)
        folium.Marker([start_lat, start_lon], icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
        m.fit_bounds([[start_lat, start_lon], [end_lat, end_lon]])

        leg = f'''<div style="position: absolute; top: 20px; right: 20px; width: 240px; background-color: rgba(9, 9, 11, 0.95); color: white; border: 1px solid #06b6d4; z-index: 9999; padding: 15px; border-radius: 10px; font-family: sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);"><h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Results</h4><div style="font-size: 12px; line-height: 1.6;"><b>AI Optimized:</b> {a_t/60:.1f} mins | {a_d/1000:.1f} km<br><b>Shortest Path:</b> {s_t/60:.1f} mins | {s_d/1000:.1f} km<br><b style="color: #22c55e;">Time Saved: {max(0, (s_t - a_t)/60.0):.1f} mins</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Traffic</b><div style="margin-bottom:4px;"><i style="background:#22c55e; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy</div></div>'''
        m.get_root().html.add_child(folium.Element(leg))
        print(f"✅ Route Calculation Time: {time.time() - start_time:.4f}s")
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)