"""
TRAFFICCONTROL: SPATIO-TEMPORAL TRAFFIC FORECASTING & PREDICTIVE ROUTING
Major Project CT707 | Department of Computer Engineering
Kathmandu Engineering College
"""

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

# --- 1. GLOBAL CONFIGURATION ---
SEQ_IN, SEQ_OUT = 12, 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)

KATHMANDU_HOTSPOTS = {
    "Kalanki": (27.6936, 85.2806, 800, 2.5),
    "Thapathali": (27.6933, 85.3168, 700, 2.5),
    "Koteshwor": (27.6745, 85.3468, 900, 2.5),
    "Chabahil": (27.7214, 85.3512, 700, 2.5),
    "Gaushala": (27.7120, 85.3468, 600, 2.5),
    "Tripureswor":(27.6953, 85.3115, 600, 2.5),
    "Thamel": (27.7145, 85.3120, 700, 1.8) 
}

# --- 2. DATASET INITIALIZATION & PREPROCESSING ---
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


# --- 3. SPATIO-TEMPORAL MODEL ARCHITECTURE ---
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
    print("✅ Advanced Temporal GCN-GRU Model Loaded.")
model.eval()

# --- 4. GEOSPATIAL GRAPH CACHING ---
CACHED_GRAPHS = {}
CITY_QUERIES = {'san_francisco': "San Francisco, California", 'kathmandu': "Kathmandu, Nepal"}


def get_city_graph(city_id):
    if city_id in CACHED_GRAPHS: return CACHED_GRAPHS[city_id]
    file_name = f"data/{city_id}_map.graphml"
    if os.path.exists(file_name):
        print(f"Loading {city_id} map from disk...")
        G = ox.load_graphml(file_name)
    else:
        print(f"Downloading {city_id} map from OSM... (Please wait)")
        ox.settings.use_cache = False 
        if city_id == 'san_francisco':
            G = ox.graph_from_point((37.7749, -122.4194), dist=20000, network_type='drive')
        else:
            G = ox.graph_from_point((27.67, 85.32), dist=12000, network_type='drive')
        ox.save_graphml(G, filepath=file_name)
        ox.settings.use_cache = True
    CACHED_GRAPHS[city_id] = G
    return G


# --- 5. API ENDPOINTS ---
@app.route('/get_overview', methods=['POST', 'GET'])
def get_overview():

    try:
        user_agent = request.headers.get('User-Agent', '').lower()
        is_mobile = 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent

        found_idx = int(T * 0.8) + 100 
        input_list = [np.stack([traffic_scaled[t], np.full(N, hours_sin[t]), np.full(N, days_sin[t])], axis=1) for t in range(found_idx, found_idx + SEQ_IN)]
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(): preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0,:].cpu().numpy().reshape(-1, 1)).flatten()
        
        # Load optimized homepage graph
        file_name = "data/sf_highway_map.graphml"
        if os.path.exists(file_name):
            G = ox.load_graphml(file_name)
        else:
            custom_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary"]'
            G = ox.graph_from_place("San Francisco, California", custom_filter=custom_filter)
            ox.save_graphml(G, filepath=file_name)

        m = folium.Map(tiles='CartoDB dark_matter', zoom_control=not is_mobile)
        edge_index = 0
        
        for u, v, _, data in G.edges(keys=True, data=True):
            hw_type = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            if hw_type in ['motorway', 'motorway_link', 'trunk', 'primary', 'secondary']:
                flow = predicted_flows[edge_index % N]
                color = '#10b981' if flow < 200 else '#f59e0b' if flow < 400 else '#ef4444'
                coords = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                folium.PolyLine(coords, color=color, weight=5, opacity=0.9).add_to(m)
                edge_index += 1
                
        m.fit_bounds([[37.7, -122.5], [37.81, -122.38]]) 
        
        if is_mobile:
            legend = '''<div style="position: absolute; bottom: 10px; right: 10px; width: 120px; background-color: rgba(9, 9, 11, 0.8); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; z-index: 999999; font-size: 10px; color: white; padding: 8px; font-family: sans-serif;"><b style="font-size:9px; color:#a1a1aa; display:block; margin-bottom:5px;">TRAFFIC</b><div style="margin-bottom:4px;"><i style="background:#10b981; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i> Free</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i> Mod</div><div><i style="background:#ef4444; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i> Heavy</div></div>'''
            m.get_root().html.add_child(folium.Element(legend))
        else:
            legend = '''<div style="position: absolute; top: 90px; left: 20px; width: 140px; background-color: rgba(9, 9, 11, 0.9); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; z-index: 999999; font-size: 12px; color: white; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:8px;">Traffic Flow</b><div style="margin-bottom:6px;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free</div><div style="margin-bottom:6px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy</div></div>'''
            m.get_root().html.add_child(folium.Element(legend))
            
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500,


@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        start_time = time.time()
        payload = request.json
        city_id = payload.get('city', 'san_francisco')
        start_lat, start_lon = float(payload['start_lat']), float(payload['start_lon'])
        end_lat, end_lon = float(payload['end_lat']), float(payload['end_lon'])
        roadblocks = payload.get('roadblocks', [])

        target_day_int = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}.get(payload.get('day', 'Friday'), 4)
        target_hour = int(payload.get('hour', 17))
        
        # 1. AI Forecast
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
        
        # 2. Load Graph and Snapping
        G_full = get_city_graph(city_id)
        
        # SNAP to nodes for validation
        n_start_full = ox.distance.nearest_nodes(G_full, start_lon, start_lat)
        n_end_full = ox.distance.nearest_nodes(G_full, end_lon, end_lat)  # NOW USED BELOW
        
        # Boundary validation: Ensure both points are within 2.5km of a road
        d_start = ox.distance.great_circle(start_lat, start_lon, G_full.nodes[n_start_full]['y'], G_full.nodes[n_start_full]['x'])
        d_end = ox.distance.great_circle(end_lat, end_lon, G_full.nodes[n_end_full]['y'], G_full.nodes[n_end_full]['x'])
        
        if d_start > 2500 or d_end > 2500:
            return jsonify({'error': "Coordinates too far from road network."}), 400

        # Truncate for speed
        dist_buffer = 0.15 
        bbox = (min(start_lon, end_lon) - dist_buffer, min(start_lat, end_lat) - dist_buffer,
                max(start_lon, end_lon) + dist_buffer, max(start_lat, end_lat) + dist_buffer)
        G = ox.truncate.truncate_graph_bbox(G_full, bbox=bbox)
        
        n_start = ox.distance.nearest_nodes(G, start_lon, start_lat)
        n_end = ox.distance.nearest_nodes(G, end_lon, end_lat)

        # Precision Roadblock Logic
        blocked_edges = set()
        for rb in roadblocks:
            try:
                u_rb, v_rb, _ = ox.distance.nearest_edges(G, float(rb['lon']), float(rb['lat']))
                blocked_edges.add((u_rb, v_rb))
            except: pass

        # --- 3. Dynamic Edge Weighting (FIXED & OPTIMIZED) ---

        # Normalize predicted flows (CRITICAL)
        predicted_flows = np.clip(predicted_flows, 50, 1200)

        avg_flow = np.mean(predicted_flows)
        trend = max(0.6, avg_flow / 250.0)
        is_night_time = not (7 <= target_hour < 22)

        for u, v, _, data in G.edges(keys=True, data=True):

            hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            is_highway = hw in ['motorway', 'trunk', 'primary', 'motorway_link']

            # --- FLOW ESTIMATION (FIXED: deterministic mapping) ---
            if city_id == 'san_francisco':
                edge_index = (u + v) % N   # ✅ deterministic (REMOVED hash randomness)
                flow = predicted_flows[edge_index]
            else:
                if is_highway and is_night_time:
                    base_flow = 150
                elif is_highway:
                    base_flow = 800
                elif hw in ['secondary', 'tertiary']:
                    base_flow = 300
                else:
                    base_flow = 120

                flow = base_flow * trend

            data['flow_val'] = flow

            # --- ROAD PARAMETERS ---
            if is_highway:
                cap, spd = 1200.0, 65.0
            else:
                cap, spd = 400.0, 35.0

            # --- BASE TIME ---
            base_time = data.get('length', 100) / (spd / 3.6)

            # --- CONGESTION MODEL (STRONG BPR FIX) ---
            ratio = min(flow / cap, 3.0)

            # Strong congestion impact
            travel_time = base_time * (1 + 1.2 * (ratio ** 4))

            # Heavy penalty when overloaded
            if ratio > 1.0:
                travel_time *= (2.0 + ratio)

            # --- ROAD TYPE PENALTY (REALISM FIX) ---
            if hw not in ['motorway', 'trunk', 'primary']:
                travel_time *= 1.3

            # --- ROADBLOCK HANDLING ---
            if (u, v) in blocked_edges or (v, u) in blocked_edges:
                data['travel_time'] = float('inf')
                data['is_blocked'] = True
            else:
                data['travel_time'] = travel_time
                data['is_blocked'] = False

            # --- STORE METRICS ---
            data['real_time'] = travel_time
            data['distance_weight'] = data.get('length', 100)
        # 4. Dijkstra Execution
        try:
            r_s = nx.shortest_path(G, n_start, n_end, weight='distance_weight')
            r_a = nx.shortest_path(G, n_start, n_end, weight='travel_time')
        except nx.NetworkXNoPath:
            return jsonify({'error': "Roadblock isolated the destination. No valid route exists."}), 400

        # Exact Geometry Drawing Fix
        def get_route_geometry(graph, route):
            coords = []
            for u_node, v_node in zip(route[:-1], route[1:]):
                edge_data = graph.get_edge_data(u_node, v_node)[0]
                if 'geometry' in edge_data:
                    coords.extend([(lat, lon) for lon, lat in list(edge_data['geometry'].coords)])
                else:
                    coords.extend([(graph.nodes[u_node]['y'], graph.nodes[u_node]['x']), (graph.nodes[v_node]['y'], graph.nodes[v_node]['x'])])
            return coords

        def get_stats(g, r):
            hit_block = any(g.get_edge_data(u_n, v_n)[0].get('is_blocked', False) for u_n, v_n in zip(r[:-1], r[1:]))
            dists = [g.get_edge_data(u_n, v_n)[0].get('length', 0) for u_n, v_n in zip(r[:-1], r[1:])]
            times = [g.get_edge_data(u_n, v_n)[0].get('real_time', 0) for u_n, v_n in zip(r[:-1], r[1:])]
            return sum(dists), (99999 if hit_block else sum(times)), hit_block

        s_d, s_t, s_blocked = get_stats(G, r_s)
        a_d, a_t, _ = get_stats(G, r_a)

        # 5. Visualization
        m = folium.Map(tiles='CartoDB dark_matter')
        route_nodes = set(r_s + r_a)
        lats = [G.nodes[n]['y'] for n in route_nodes]
        lons = [G.nodes[n]['x'] for n in route_nodes]
        min_la, max_la, min_lo, max_lo = min(lats) - 0.015, max(lats) + 0.015, min(lons) - 0.015, max(lons) + 0.015

        for u, v, _, data in G.edges(keys=True, data=True):
            la_u, lo_u = G.nodes[u]['y'], G.nodes[u]['x']
            if min_la <= la_u <= max_la and min_lo <= lo_u <= max_lo:
                hw_t = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
                if hw_t in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']:
                    val = data['flow_val']
                    clr = '#22c55e' if val < 250 else '#f59e0b' if val < 550 else '#ef4444'
                    c = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                    folium.PolyLine(c, color=clr, weight=3, opacity=0.4).add_to(m)

        folium.PolyLine(get_route_geometry(G, r_a), color='#06b6d4', weight=6, opacity=1.0).add_to(m)
        folium.PolyLine(get_route_geometry(G, r_s), color='#FFFFFF', weight=3, opacity=0.7, dash_array='10, 15').add_to(m)
        folium.Marker([G.nodes[n_start]['y'], G.nodes[n_start]['x']], icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
        folium.Marker([G.nodes[n_end]['y'], G.nodes[n_end]['x']], icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
        
        for rb in roadblocks:
            folium.Marker([rb['lat'], rb['lon']], icon=folium.Icon(color='orange', icon='triangle-exclamation', prefix='fa')).add_to(m)

        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        s_d, s_t, s_blocked = get_stats(G, r_s)
        a_d, a_t, _ = get_stats(G, r_a)

        # --- SMART TIME COMPARISON ---
        time_diff = (s_t - a_t) / 60.0

        if s_blocked:
            saved_disp = "Route Saved (Avoided Blockage)"
            s_t_disp = "Impassable"
        else:
            s_t_disp = f"{s_t/60:.1f} mins"
            if time_diff > 0:
                saved_disp = f"{time_diff:.1f} mins faster"
            else:
                saved_disp = f"{abs(time_diff):.1f} mins slower"
        
        leg = f'''<div style="position: absolute; top: 20px; right: 20px; width: 240px; background-color: rgba(9, 9, 11, 0.95); color: white; border: 1px solid #06b6d4; z-index: 9999; padding: 15px; border-radius: 10px; font-family: sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);"><h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Results</h4><div style="font-size: 12px; line-height: 1.6;"><b>AI Optimized:</b> {a_t/60:.1f} mins | {a_d/1000:.1f} km<br><b>Shortest Path:</b> {s_t_disp} | {s_d/1000:.1f} km<br><b style="color: #22c55e;">Time Saved: {saved_disp}</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Traffic</b><div style="margin-bottom:4px;"><i style="background:#22c55e; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy</div></div>'''
        m.get_root().html.add_child(folium.Element(leg))
        
        print(f"[PERFORMANCE] Total Route Calculation Time: {time.time() - start_time:.4f} seconds")
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_isochrone', methods=['POST'])
def get_isochrone():
    try:
        start_time = time.time()
        payload = request.json
        mode = payload.get('mode', 'custom') 
        time_limit_mins = float(payload.get('time_limit', 10))
        time_limit_sec = time_limit_mins * 60
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
        
        G = get_city_graph('kathmandu')
        avg_flow = np.mean(predicted_flows)
        trend = np.clip(avg_flow / 300.0, 0.7, 2.0)
        is_night_time = not (7 <= target_hour < 22)

        for u, v, _, data in G.edges(keys=True, data=True):
            hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            is_h = hw in ['motorway', 'trunk', 'primary', 'secondary']
            
            if is_h and is_night_time: base_flow = 150
            elif is_h: base_flow = 800
            elif hw in ['tertiary']: base_flow = 300
            else: base_flow = 100
            
            f = base_flow * trend * (0.8 + 0.4 * ((u % 10) / 10.0))
            data['flow_val'] = f
            manual_penalty = 1.0
            if not is_night_time:
                node_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
                node_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
                for _, (h_lat, h_lon, radius, penalty) in KATHMANDU_HOTSPOTS.items():
                    if ox.distance.great_circle(node_lat, node_lon, h_lat, h_lon) <= radius:
                        manual_penalty = penalty
                        break
                    
            cap, spd = (1200.0, 60.0) if is_h else (400.0, 40.0)
            ratio = data['flow_val'] / cap
            
            tm = (data.get('length', 100) / (spd / 3.6)) * (1 + 0.15 * (ratio) ** 4)
            ratio = min(data['flow_val'] / cap, 1.5)
            tm = (data.get('length', 100) / (spd / 3.6)) * (1 + 0.15 * (ratio ** 4))
            congestion_penalty = 1 + 0.25 * ratio
            data['travel_time'] = tm * congestion_penalty * manual_penalty
        reachable_nodes, sources = set(), []
        
        if mode == 'hospitals':
            hospitals = [{'name': 'Bir Hospital', 'lat': 27.7056, 'lon': 85.3137}, {'name': 'TU Teaching', 'lat': 27.7353, 'lon': 85.3310}, {'name': 'Civil Service', 'lat': 27.6863, 'lon': 85.3388}, {'name': 'Grande Int.', 'lat': 27.7528, 'lon': 85.3258}, {'name': 'Norvic Int.', 'lat': 27.6899, 'lon': 85.3189}, {'name': 'HAMS', 'lat': 27.7332, 'lon': 85.3457}, {'name': 'ERA Int.', 'lat': 27.7193, 'lon': 85.3091}, {'name': 'Nepal-Bharat Maitri', 'lat': 27.7116, 'lon': 85.3454}, {'name': 'Grande City', 'lat': 27.7111, 'lon': 85.3148}, {'name': 'All Nepal', 'lat': 27.7330, 'lon': 85.3146}, {'name': 'CIWEC', 'lat': 27.7204, 'lon': 85.3177}, {'name': 'Annapurna Neuro', 'lat': 27.701, 'lon': 85.324}, {'name': 'Green City', 'lat': 27.735, 'lon': 85.350}, {'name': 'Manmohan Memorial', 'lat': 27.735, 'lon': 85.300}, {'name': 'KIST Medical', 'lat': 27.658, 'lon': 85.324}, {'name': 'Overseas Friendship', 'lat': 27.714, 'lon': 85.312}]
            for h in hospitals:
                try:
                    node = ox.distance.nearest_nodes(G, h['lon'], h['lat'])
                    sources.append((h['lat'], h['lon'], h['name'], node))
                    subgraph = nx.ego_graph(G, node, radius=time_limit_sec, distance='travel_time')
                    reachable_nodes.update(subgraph.nodes())
                except: pass
        else:
            start_lat, start_lon = float(payload['start_lat']), float(payload['start_lon'])
            node = ox.distance.nearest_nodes(G, start_lon, start_lat)
            sources.append((start_lat, start_lon, "Dispatch Point", node))
            subgraph = nx.ego_graph(G, node, radius=time_limit_sec, distance='travel_time')
            reachable_nodes.update(subgraph.nodes())
        
        center_lat = sum(s[0] for s in sources) / len(sources) if sources else 27.7172
        center_lon = sum(s[1] for s in sources) / len(sources) if sources else 85.3240
        
        user_agent = request.headers.get('User-Agent', '').lower()
        is_mobile = 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB dark_matter', zoom_control=not is_mobile)
        
        for u, v, _, data in G.edges(keys=True, data=True):
            if 'geometry' in data: coords = [(lat, lon) for lon, lat in list(data['geometry'].coords)]
            else: coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            if u in reachable_nodes and v in reachable_nodes:
                val = data.get('flow_val', 0)
                clr = '#10b981' if val < 250 else '#f59e0b' if val < 550 else '#ef4444'
                folium.PolyLine(coords, color=clr, weight=1.5, opacity=0.7).add_to(m)
            else:
                hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
                if hw in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']:
                    folium.PolyLine(coords, color='#333333', weight=1, opacity=0.3).add_to(m)

        for lat, lon, name, _ in sources:
            icon_color = 'red' if mode == 'hospitals' else 'blue'
            icon_shape = 'plus' if mode == 'hospitals' else 'truck-medical'
            folium.Marker([lat, lon], popup=name, icon=folium.Icon(color=icon_color, icon=icon_shape, prefix='fa')).add_to(m)

        coverage_pct = (len(reachable_nodes) / len(G.nodes)) * 100
        
        if is_mobile:
            leg = f'''<div style="position: absolute; bottom: 10px; right: 10px; width: 170px; background-color: rgba(9, 9, 11, 0.85); color: white; border: 1px solid #06b6d4; z-index: 9999; padding: 10px; border-radius: 8px; font-family: sans-serif; font-size: 10px;"><h4 style="margin: 0 0 5px 0; color: #06b6d4; font-size:11px;">Isochrone Analysis</h4><div style="line-height: 1.4;"><b>Limit:</b> {time_limit_mins}m<br><b>Nodes:</b> {len(reachable_nodes)}<br><b style="color: #10b981;">Coverage: {coverage_pct:.1f}%</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 6px 0;"><div style="margin-bottom:3px;"><i style="background:#10b981; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i> Free</div><div style="margin-bottom:3px;"><i style="background:#f59e0b; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i> Mod</div><div><i style="background:#ef4444; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i> Heavy</div></div>'''
        else:
            leg = f'''<div style="position: absolute; top: 20px; right: 20px; width: 260px; background-color: rgba(9, 9, 11, 0.95); color: white; border: 1px solid #06b6d4; z-index: 9999; padding: 15px; border-radius: 10px; font-family: sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);"><h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Isochrone Analysis</h4><div style="font-size: 12px; line-height: 1.6;"><b>Time Limit:</b> {time_limit_mins} Minutes<br><b>Intersections Reached:</b> {len(reachable_nodes)}<br><b style="color: #10b981;">Coverage: {coverage_pct:.1f}%</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Traffic within Reachable Zone</b><div style="margin-bottom:4px;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; float:left; margin-right:8px; margin-top:2px;"></i> Free Flow</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; float:left; margin-right:8px; margin-top:2px;"></i> Moderate Traffic</div><div style="margin-bottom:8px;"><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; float:left; margin-right:8px; margin-top:2px;"></i> Severe Congestion</div><div><i style="background:#333333; width:15px; height:2px; float:left; margin-right:6px; margin-top:6px;"></i> Unreachable in {time_limit_mins} mins</div></div>'''
            
        m.get_root().html.add_child(folium.Element(leg))
        
        print(f"\n\n[PERFORMANCE] Isochrone Calculation Time: {time.time() - start_time:.4f} seconds\n")
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
