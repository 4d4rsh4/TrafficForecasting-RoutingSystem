import time
import math
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
DEVICE = torch.device('cpu')

app = Flask(__name__)
CORS(app)

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
except Exception as e:
    print(f"❌ Data Error: {e}")
    T, N = 1000, 307
    adj = torch.eye(N)


class GraphConvLayer(nn.Module):

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feat, out_feat))
        self.bias = nn.Parameter(torch.zeros(out_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return torch.matmul(torch.matmul(adj, x), self.weight) + self.bias


class TemporalGCNGRU(nn.Module):

    def __init__(self, num_sensors, gcn_hid, gru_hid, seq_in, seq_out, adj):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_out = seq_out
        self.seq_in = seq_in
        self.adj = adj
        self.gcn1 = GraphConvLayer(3, gcn_hid) 
        self.gcn2 = GraphConvLayer(gcn_hid, gcn_hid)
        self.gru = nn.GRU(gcn_hid, gru_hid, num_layers=1, batch_first=True)
        self.fc = nn.Linear(gru_hid, self.seq_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, L, N, _ = x.shape
        gcn_out = [self.gcn2(self.relu(self.gcn1(x[:, t,:,:], self.adj)), self.adj) for t in range(L)]
        gcn_seq = torch.stack(gcn_out, dim=1)
        _, h = self.gru(gcn_seq.view(B * N, L, -1))
        out = self.fc(h.squeeze(0)) 
        return out.view(B, N, self.seq_out).permute(0, 2, 1) 


model = TemporalGCNGRU(num_sensors=N, gcn_hid=32, gru_hid=64, seq_in=SEQ_IN, seq_out=SEQ_OUT, adj=adj.to(DEVICE))
if os.path.exists('traffic_model_temporal.pth'):
    model.load_state_dict(torch.load('traffic_model_temporal.pth', map_location=DEVICE))
model.eval()

CACHED_GRAPHS = {}
CITY_QUERIES = {'san_francisco': "San Francisco, California", 'kathmandu': "Kathmandu, Nepal"}


def get_city_graph(city_id):
    if city_id in CACHED_GRAPHS: return CACHED_GRAPHS[city_id]
    file_name = f"{city_id}_map.graphml"
    if os.path.exists(file_name):
        print(f"Loading {city_id} map from disk...")
        G = ox.load_graphml(file_name)
    else:
        print(f"Downloading {city_id} map... (This takes 30-60s)")
        G = ox.graph_from_place(CITY_QUERIES.get(city_id, "San Francisco, California"), network_type='drive')
        ox.save_graphml(G, filepath=file_name)
    CACHED_GRAPHS[city_id] = G
    return G


@app.route('/get_overview', methods=['POST'])
def get_overview():
    try:
        found_idx = int(T * 0.8) + 100 
        input_list = [np.stack([traffic_scaled[t], np.full(N, hours_sin[t]), np.full(N, days_sin[t])], axis=1) for t in range(found_idx, found_idx + SEQ_IN)]
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(): preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0,:].cpu().numpy().reshape(-1, 1)).flatten()

        G = get_city_graph('san_francisco')
        m = folium.Map(tiles='CartoDB dark_matter')
        
        edge_index = 0
        for u, v, _, data in G.edges(keys=True, data=True):
            hw_type = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            if hw_type in ['motorway', 'motorway_link', 'trunk']:
                flow = predicted_flows[edge_index % N]
                color = '#10b981' if flow < 200 else '#f59e0b' if flow < 400 else '#ef4444'
                coords = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)
                edge_index += 1

        m.fit_bounds([[37.7, -122.5], [37.8, -122.3]]) 
        legend = '''<div style="position: absolute; top: 90px; left: 20px; width: 140px; background-color: rgba(9, 9, 11, 0.9); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; z-index: 999999; font-size: 12px; color: white; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:8px;">Traffic Flow</b><div style="margin-bottom:6px;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free</div><div style="margin-bottom:6px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy</div></div>'''
        m.get_root().html.add_child(folium.Element(legend))
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        start_time=time.time()
        payload = request.json
        city_id = payload.get('city', 'san_francisco')
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

        G_full = get_city_graph(city_id)
        dist_buffer = 0.08 
        bbox = (min(start_lon, end_lon) - dist_buffer, min(start_lat, end_lat) - dist_buffer,
                max(start_lon, end_lon) + dist_buffer, max(start_lat, end_lat) + dist_buffer)
        G = ox.truncate.truncate_graph_bbox(G_full, bbox=bbox)
        
        avg_network_flow = np.mean(predicted_flows)
        temporal_trend = max(0.5, avg_network_flow / 200.0) 

        for u, v, _, data in G.edges(keys=True, data=True):
            highway_type = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            is_highway = highway_type in ['motorway', 'trunk', 'primary', 'motorway_link']
            
            if city_id == 'san_francisco':
                flow = predicted_flows[(u + v) % N]
            else:
                if is_highway: base_flow = 800
                elif highway_type in ['secondary', 'tertiary']: base_flow = 300
                else: base_flow = 100
                geo_noise = 0.8 + 0.4 * ((u % 10) / 10.0)
                flow = base_flow * temporal_trend * geo_noise

            data['flow_val'] = flow * 1.2 if is_highway else flow * 0.4
            capacity, speed = (1200.0, 65.0) if is_highway else (400.0, 35.0)
            
            congestion_ratio = data['flow_val'] / capacity
            physical_time = (data.get('length', 100) / (speed / 3.6)) * (1 + 1.5 * (congestion_ratio) ** 4)
            comfort_penalty = 5.0 if congestion_ratio >= 0.7 else 2.0 if congestion_ratio >= 0.4 else 1.0

            data['travel_time'] = physical_time * comfort_penalty
            data['distance_weight'] = data.get('length', 100)

        n_start = ox.distance.nearest_nodes(G, start_lon, start_lat)
        n_end = ox.distance.nearest_nodes(G, end_lon, end_lat)

        def calculate_dist(lat1, lon1, node_id, graph):
            lat2 = graph.nodes[node_id]['y']
            lon2 = graph.nodes[node_id]['x']
            
            R = 6371000
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            
            a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
            return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

        dist_to_start = calculate_dist(start_lat, start_lon, n_start, G)
        dist_to_end = calculate_dist(end_lat, end_lon, n_end, G)

        if dist_to_start > 1000 or dist_to_end > 1000:
            return jsonify({
                'error': "Coordinates are too far from a valid road network. Please ensure your markers are placed within the city limits and not in water bodies."
            }), 400

        try:
            route_shortest = nx.shortest_path(G, n_start, n_end, weight='distance_weight')
            route_ai = nx.shortest_path(G, n_start, n_end, weight='travel_time')
        except nx.NetworkXNoPath:
            return jsonify({'error': "No valid driving path found between these two points. Please move the markers closer to main roads."}), 400

        def get_route_stats(graph, route):
            dists, times = zip(*[(graph.get_edge_data(u, v)[0].get('length', 0), graph.get_edge_data(u, v)[0].get('travel_time', 0)) for u, v in zip(route[:-1], route[1:])])
            return sum(dists), sum(times)

        s_dist_m, s_time_s = get_route_stats(G, route_shortest)
        a_dist_m, a_time_s = get_route_stats(G, route_ai)

        m = folium.Map(tiles='CartoDB dark_matter')
        for u, v, _, data in G.edges(keys=True, data=True):
            f = data['flow_val']
            color = '#22c55e' if f < 250 else '#f59e0b' if f < 550 else '#ef4444'
            coords = [[lat, lon] for lon, lat in data['geometry'].coords] if 'geometry' in data else [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
            folium.PolyLine(coords, color=color, weight=2, opacity=0.3).add_to(m)

        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in route_ai], color='#06b6d4', weight=8, opacity=1.0).add_to(m)
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in route_shortest], color='#FFFFFF', weight=3, opacity=1.0, dash_array='10, 15').add_to(m)
        folium.Marker([start_lat, start_lon], icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
        m.fit_bounds([[start_lat, start_lon], [end_lat, end_lon]])

        stats_html = f'''<div style="position: absolute; top: 20px; right: 20px; width: 240px; background-color: rgba(15, 23, 42, 0.95); color: white; border: 2px solid #06b6d4; z-index: 9999; padding: 15px; border-radius: 10px; font-family: 'Inter', sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);"><h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Simulation Results</h4><div style="font-size: 12px; line-height: 1.6;"><b>AI Optimized:</b> {a_time_s/60:.1f} mins | {a_dist_m/1000:.1f} km<br><b>Shortest Path:</b> {s_time_s/60:.1f} mins | {s_dist_m/1000:.1f} km<br><b style="color: #22c55e;">Time Saved: {max(0, (s_time_s - a_time_s)/60.0):.1f} mins</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Predicted Traffic</b><div style="margin-bottom:4px;"><i style="background:#22c55e; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free Flow</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy Congestion</div></div>'''
        m.get_root().html.add_child(folium.Element(stats_html))
        total_time = time.time() - start_time
        print(f"\n [PERFORMANCE] Total Route Calculation Time: {total_time:.4f} seconds\n")
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)