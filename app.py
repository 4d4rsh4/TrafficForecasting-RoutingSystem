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
    print(f"Data Loaded: {N} sensors on {DEVICE}.")
except Exception as e:
    print(f"Data Error: {e}")
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
    print("Advanced Model Loaded.")
model.eval()

# --- 4. MAP CACHING ---
CACHED_GRAPHS = {}
CITY_QUERIES = {'san_francisco': "San Francisco, California"}

def get_city_graph(city_id):
    if city_id in CACHED_GRAPHS: return CACHED_GRAPHS[city_id]
    file_name = f"{city_id}_map.graphml"
    
    if os.path.exists(file_name):
        print(f"Loading {city_id} map from disk...")
        G = ox.load_graphml(file_name)
    else:
        print(f"Downloading expanded {city_id} map... (This may take 1-2 minutes)")
        
        # FIX: Temporarily disable OSMnx caching to force a fresh download
        ox.settings.use_cache = False 
        
        if city_id == 'san_francisco':
            # 20km radius: Covers SF, Oakland, and South SF
            G = ox.graph_from_point((37.7749, -122.4194), dist=20000, network_type='drive')
        else:
            # FIX: Massive 12km radius from the center of Patan (Lalitpur)
            # This forces the map to pull everything from Kathmandu down to Godawari.
            G = ox.graph_from_point((27.67, 85.32), dist=12000, network_type='drive')
            
        ox.save_graphml(G, filepath=file_name)
        
        # Turn caching back on for future operations
        ox.settings.use_cache = True
        print(f"✅ {city_id} map downloaded and saved.")
        
    CACHED_GRAPHS[city_id] = G
    return G


@app.route('/get_overview', methods=['POST', 'GET'])
def get_overview():
    try:
        found_idx = int(T * 0.8) + 100 
        input_list = [np.stack([traffic_scaled[t], np.full(N, hours_sin[t]), np.full(N, days_sin[t])], axis=1) for t in range(found_idx, found_idx + SEQ_IN)]
        input_tensor = torch.FloatTensor(np.array(input_list)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): preds = model(input_tensor)
        predicted_flows = scaler.inverse_transform(preds[0, 0,:].cpu().numpy().reshape(-1, 1)).flatten()
        
        file_name = "sf_highway_map.graphml"
        if os.path.exists(file_name):
            G = ox.load_graphml(file_name)
        else:
            print("Downloading HIGHWAY-ONLY map for San Francisco...")
            custom_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary"]'
            G = ox.graph_from_place("San Francisco, California", custom_filter=custom_filter)
            ox.save_graphml(G, filepath=file_name)

        m = folium.Map(tiles='CartoDB dark_matter')
        
        edge_index = 0
        for u, v, _, data in G.edges(keys=True, data=True):
            flow = predicted_flows[edge_index % N]
            color = '#10b981' if flow < 200 else '#f59e0b' if flow < 400 else '#ef4444'
            
            weight = 5
            
            coords = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            folium.PolyLine(coords, color=color, weight=weight, opacity=0.9).add_to(m)
            edge_index += 1

        m.fit_bounds([[37.7, -122.5], [37.81, -122.38]]) 
        
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
        
        print(f"\n--- Dynamic Map Fetch for route: ({start_lat},{start_lon}) -> ({end_lat},{end_lon}) ---")
        
        G = get_city_graph(city_id) # Use the full cached graph directly! (Do not truncate)

        n_start = ox.distance.nearest_nodes(G, start_lon, start_lat)
        n_end = ox.distance.nearest_nodes(G, end_lon, end_lat)

        def calculate_dist(l1, ln1, nid, g):
            return ox.distance.great_circle(l1, ln1, g.nodes[nid]['y'], g.nodes[nid]['x'])

        if calculate_dist(start_lat, start_lon, n_start, G) > 2500 or calculate_dist(end_lat, end_lon, n_end, G) > 2500:
            return jsonify({'error': "Coordinates too far from road network. Please place markers closer to land."}), 400

        # Apply weights to the full graph (Numpy makes this very fast)
        avg_flow = np.mean(predicted_flows)
        trend = max(0.5, avg_flow / 200.0) 

        for u, v, _, data in G.edges(keys=True, data=True):
            hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            is_h = hw in ['motorway', 'trunk', 'primary', 'motorway_link']
            
            if city_id == 'san_francisco':
                f = predicted_flows[(u + v) % N]
            else:
                base = 800 if is_h else 300 if hw in ['secondary', 'tertiary'] else 100
                f = base * trend * (0.8 + 0.4 * ((u % 10) / 10.0))
            
            data['flow_val'] = f * 1.2 if is_h else f * 0.4
            cap, spd = (1200.0, 65.0) if is_h else (400.0, 35.0)
            ratio = data['flow_val'] / cap
            
            tm = (data.get('length', 100) / (spd / 3.6)) * (1 + 1.5 * (ratio) ** 4)
            pen = 5.0 if ratio >= 0.7 else 2.0 if ratio >= 0.4 else 1.0
            
            data['travel_time'] = tm * pen
            data['distance_weight'] = data.get('length', 100)

        # Run Dijkstra on the full graph
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

        # --- UI OPTIMIZATION: DRAW ONLY VISIBLE AREA ---
        m = folium.Map(tiles='CartoDB dark_matter')
        
        # Calculate a tight bounding box strictly around the generated routes
        route_nodes = set(r_s + r_a)
        lats = [G.nodes[n]['y'] for n in route_nodes]
        lons = [G.nodes[n]['x'] for n in route_nodes]
        min_lat, max_lat = min(lats) - 0.015, max(lats) + 0.015
        min_lon, max_lon = min(lons) - 0.015, max(lons) + 0.015
        
        major_roads = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link']
        
        for u, v, _, data in G.edges(keys=True, data=True):
            lat_u, lon_u = G.nodes[u]['y'], G.nodes[u]['x']
            
            # ONLY render if the road is physically inside the viewable camera box!
            if min_lat <= lat_u <= max_lat and min_lon <= lon_u <= max_lon:
                hw_type = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
                if hw_type in major_roads:
                    val = data['flow_val']
                    clr = '#22c55e' if val < 250 else '#f59e0b' if val < 550 else '#ef4444'
                    c = [(lat, lon) for lon, lat in list(data['geometry'].coords)] if 'geometry' in data else [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                    folium.PolyLine(c, color=clr, weight=3, opacity=0.5).add_to(m)

        # Draw Paths and Markers
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in r_a], color='#06b6d4', weight=6, opacity=1.0).add_to(m)
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in r_s], color='#FFFFFF', weight=3, opacity=0.7, dash_array='10, 15').add_to(m)
        folium.Marker([start_lat, start_lon], icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        leg = f'''<div style="position: absolute; top: 20px; right: 20px; width: 240px; background-color: rgba(9, 9, 11, 0.95); color: white; border: 1px solid #06b6d4; z-index: 9999; padding: 15px; border-radius: 10px; font-family: sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);"><h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Results</h4><div style="font-size: 12px; line-height: 1.6;"><b>AI Optimized:</b> {a_t/60:.1f} mins | {a_d/1000:.1f} km<br><b>Shortest Path:</b> {s_t/60:.1f} mins | {s_d/1000:.1f} km<br><b style="color: #22c55e;">Time Saved: {max(0, (s_t - a_t)/60.0):.1f} mins</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Traffic</b><div style="margin-bottom:4px;"><i style="background:#22c55e; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Free Flow</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Moderate</div><div><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px;"></i> Heavy</div></div>'''
        m.get_root().html.add_child(folium.Element(leg))
        
        print(f"✅ Route Calculation Time: {time.time() - start_time:.4f}s")
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 8. DYNAMIC ISOCHRONE MAP ENDPOINT (EMERGENCY REACHABILITY) ---
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
        
        # 1. Query AI Model
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
        
        # 2. Load Kathmandu Map
        G = get_city_graph('kathmandu')
        
        avg_flow = np.mean(predicted_flows)
        trend = max(0.5, avg_flow / 200.0) 

        # 3. APPLY AI TRAFFIC WEIGHTS
        for u, v, _, data in G.edges(keys=True, data=True):
            hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
            is_h = hw in ['motorway', 'trunk', 'primary', 'secondary']
            
            base = 800 if is_h else 300 if hw in ['tertiary'] else 100
            f = base * trend * (0.8 + 0.4 * ((u % 10) / 10.0))
            
            data['flow_val'] = f
            cap, spd = (1200.0, 60.0) if is_h else (400.0, 40.0)
            ratio = data['flow_val'] / cap
            
            tm = (data.get('length', 100) / (spd / 3.6)) * (1 + 1.5 * (ratio) ** 4)
            pen = 4.0 if ratio >= 0.7 else 1.5 if ratio >= 0.4 else 1.0
            data['travel_time'] = tm * pen

        # 4. CALCULATE ISOCHRONE (RADIAL DIJKSTRA)
        reachable_nodes = set()
        sources = []

        if mode == 'hospitals':
            hospitals = [
                {'name': 'Bir Hospital', 'lat': 27.7056125, 'lon': 85.3137698},
                {'name': 'TU Teaching Hospital', 'lat': 27.7353768, 'lon': 85.3310225},
                {'name': 'Civil Service Hospital', 'lat': 27.6863031, 'lon': 85.3388038},
                {'name': 'Grande International', 'lat': 27.7528760, 'lon': 85.3258890},
                {'name': 'Norvic International', 'lat': 27.6899400, 'lon': 85.3189355},
                {'name': 'HAMS Hospital', 'lat': 27.7332085, 'lon': 85.3457863},
                {'name': 'ERA International', 'lat': 27.7193177, 'lon': 85.3091199},
                {'name': 'Nepal-Bharat Maitri', 'lat': 27.7116350, 'lon': 85.3454864},
                {'name': 'Grande City', 'lat': 27.7111011, 'lon': 85.3148062},
                {'name': 'All Nepal Hospital', 'lat': 27.7330454, 'lon': 85.3146717},
                {'name': 'CIWEC Hospital', 'lat': 27.7204000, 'lon': 85.3177390},
                {'name': 'Annapurna Neurological', 'lat': 27.701, 'lon': 85.324},
                {'name': 'Green City Hospital', 'lat': 27.735, 'lon': 85.350},
                {'name': 'Manmohan Memorial', 'lat': 27.735, 'lon': 85.300},
                {'name': 'KIST Medical College', 'lat': 27.658, 'lon': 85.324},
                {'name': 'Overseas Friendship Int.', 'lat': 27.714, 'lon': 85.312}
            ]
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
            sources.append((start_lat, start_lon, "Custom Dispatch Point", node))
            subgraph = nx.ego_graph(G, node, radius=time_limit_sec, distance='travel_time')
            reachable_nodes.update(subgraph.nodes())

        # 5. GENERATE MAP
        center_lat = sum(s[0] for s in sources) / len(sources) if sources else 27.7172
        center_lon = sum(s[1] for s in sources) / len(sources) if sources else 85.3240
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13 if mode == 'hospitals' else 14, tiles='CartoDB dark_matter')
        
        # --- NEW LOGIC: Color Isochrone by Traffic Congestion ---
        for u, v, _, data in G.edges(keys=True, data=True):
            if 'geometry' in data:
                coords = [(lat, lon) for lon, lat in list(data['geometry'].coords)]
            else:
                coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                
            if u in reachable_nodes and v in reachable_nodes:
                # IT IS REACHABLE: Color based on AI flow prediction
                val = data.get('flow_val', 0)
                clr = '#10b981' if val < 250 else '#f59e0b' if val < 550 else '#ef4444'
                # FIXED: Thinner lines (1.5) and slightly transparent (0.7) for less clutter
                folium.PolyLine(coords, color=clr, weight=1.5, opacity=0.7).add_to(m)
            else:
                # UNREACHABLE: Draw faint background map
                hw = data.get('highway', [''])[0] if isinstance(data.get('highway'), list) else data.get('highway', '')
                if hw in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']:
                    folium.PolyLine(coords, color='#333333', weight=1, opacity=0.3).add_to(m)

        for lat, lon, name, _ in sources:
            icon_color = 'red' if mode == 'hospitals' else 'blue'
            icon_shape = 'plus' if mode == 'hospitals' else 'truck-medical'
            folium.Marker([lat, lon], popup=name, icon=folium.Icon(color=icon_color, icon=icon_shape, prefix='fa')).add_to(m)

        coverage_pct = (len(reachable_nodes) / len(G.nodes)) * 100
        leg = f'''<div style="position: absolute; top: 20px; right: 20px; width: 260px; background-color: rgba(9, 9, 11, 0.95); color: white; border: 1px solid #06b6d4; z-index: 9999; padding: 15px; border-radius: 10px; font-family: sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.8);"><h4 style="margin: 0 0 10px 0; color: #06b6d4; font-size:14px;">Isochrone Analysis</h4><div style="font-size: 12px; line-height: 1.6;"><b>Time Limit:</b> {time_limit_mins} Minutes<br><b>Intersections Reached:</b> {len(reachable_nodes)}<br><b style="color: #10b981;">City Coverage: {coverage_pct:.1f}%</b></div><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;"><b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px;">Traffic within Reachable Zone</b><div style="margin-bottom:4px;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; float:left; margin-right:8px; margin-top:2px;"></i> Free Flow</div><div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; float:left; margin-right:8px; margin-top:2px;"></i> Moderate Traffic</div><div style="margin-bottom:8px;"><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; float:left; margin-right:8px; margin-top:2px;"></i> Severe Congestion</div><div><i style="background:#333333; width:15px; height:2px; float:left; margin-right:6px; margin-top:6px;"></i> Unreachable in {time_limit_mins} mins</div></div>'''
        m.get_root().html.add_child(folium.Element(leg))
        
        print(f"✅ Isochrone Calculation Time: {time.time() - start_time:.4f}s")
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True, port=5000)