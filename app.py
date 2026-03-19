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

# --- 1. SETUP FLASK APP ---
app = Flask(__name__)
CORS(app) 

# --- 2. LOAD PRE-TRAINED MODEL ---
class TrafficPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_steps, num_sensors):
        super(TrafficPredictor, self).__init__()
        self.output_steps = output_steps
        self.num_sensors = num_sensors
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_sensors * output_steps)

    def forward(self, x):
        _, hsn = self.gru(x)
        out = self.fc(hsn[-1])
        return out.view(x.shape[0], self.output_steps, self.num_sensors)

model = TrafficPredictor(input_dim=309, hidden_dim=64, num_layers=2, output_steps=12, num_sensors=307)
model.load_state_dict(torch.load('traffic_model.pth'))
model.eval()

# --- 3. LOAD & PREPARE DATA ---
raw_data = np.load('data/PeMS/pems04.npz')['data']
data = raw_data[:, :, 0]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

time_index = pd.date_range("2018-01-01 00:00:00", periods=data.shape[0], freq='5min')
hours_scaled = time_index.hour.values.reshape(-1, 1) / 23.0
days_scaled = time_index.dayofweek.values.reshape(-1, 1) / 6.0
data_with_time = np.hstack([data_scaled, hours_scaled, days_scaled])

# --- 4. GLOBAL GRAPH CACHING (FIXES EXTREME LOAD TIMES) ---
print("Downloading Road Network Graph (This happens ONLY ONCE on startup. Please wait 10-20 seconds)...")
# Bounding box encompassing Minneapolis to St. Paul (I-94 Corridor)
north, south, east, west = 44.995, 44.930, -93.080, -93.300
I94_BBOX = (west, south, east, north)
GLOBAL_GRAPH = ox.graph_from_bbox(bbox=I94_BBOX, network_type='drive')
print("Graph loaded successfully! Server is ready.")

# --- 5. OVERVIEW MAP ENDPOINT ---
@app.route('/get_overview_map', methods=['GET'])
def get_overview_map():
    try:
        split_idx = int(0.8 * len(time_index))
        found_idx = -1
        for i in range(split_idx, len(time_index)-12):
            if time_index[i].dayofweek == 4 and time_index[i].hour == 17:
                found_idx = i
                break
        if found_idx == -1: found_idx = split_idx + 50
        
        context_window = data_with_time[found_idx - 12 : found_idx, :]
        window_tensor = torch.FloatTensor(context_window).unsqueeze(0)
        
        with torch.no_grad(): prediction_scaled = model(window_tensor)
        
        pred_t1_scaled = prediction_scaled[0, 0, :].numpy() 
        pred_t1_real_flow = scaler.inverse_transform(pred_t1_scaled.reshape(-1, 1)).flatten()
        
        # USE CACHED GRAPH
        G = GLOBAL_GRAPH.copy()
        
        # --- MAP GENERATION ---
        m = folium.Map(tiles='CartoDB dark_matter')
        edge_index, num_sensors = 0, len(pred_t1_real_flow)
        
        for u, v, key, data_edge in G.edges(keys=True, data=True):
            highway_type = data_edge.get('highway', '')
            if isinstance(highway_type, list): highway_type = highway_type[0]
            
            if highway_type in ['motorway', 'motorway_link', 'trunk']:
                flow = pred_t1_real_flow[edge_index % num_sensors]
                if flow < 150: color = '#10b981'
                elif flow < 300: color = '#f59e0b'
                else: color = '#ef4444'
                    
                if 'geometry' in data_edge:
                    coords = [(lat, lon) for lon, lat in list(data_edge['geometry'].coords)]
                else:
                    coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                    
                folium.PolyLine(locations=coords, color=color, weight=4, opacity=0.8).add_to(m)
                edge_index += 1
        
        # Auto-frame the map
        m.fit_bounds([(south, west), (north, east)])
        
        # --- ONLY ONE LEGEND (TOP LEFT) ---
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- 6. SIMULATION API ENDPOINT ---
@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        data_input = request.json
        start_coords = (float(data_input['start_lat']), float(data_input['start_lon']))
        end_coords = (float(data_input['end_lat']), float(data_input['end_lon']))
        target_day_str = data_input['day']
        target_hour = int(data_input['hour'])

        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        target_day_int = day_map[target_day_str]
        split_idx = int(0.8 * len(time_index)) 
        found_idx = -1
        
        for i in range(split_idx, len(time_index) - 12):
            if time_index[i].dayofweek == target_day_int and time_index[i].hour == target_hour:
                found_idx = i
                break
        if found_idx == -1: found_idx = split_idx + 50 

        context_window = data_with_time[found_idx - 12 : found_idx, :]
        window_tensor = torch.FloatTensor(context_window).unsqueeze(0)

        with torch.no_grad(): prediction_scaled = model(window_tensor)
        pred_t1_scaled = prediction_scaled[0, 0, :].numpy() 
        pred_t1_real_flow = scaler.inverse_transform(pred_t1_scaled.reshape(-1, 1)).flatten()
        
        model_multipliers = []
        for flow in pred_t1_real_flow:
            factor = max(1.0, 1.0 + (flow - 100) / 100.0) 
            factor = min(factor, 4.0) 
            model_multipliers.append(factor)

        # USE CACHED GRAPH
        G = GLOBAL_GRAPH.copy()
        
        heatmap_lines = [] 
        edge_index, num_sensors = 0, len(model_multipliers)

        for u, v, key, data_edge in G.edges(keys=True, data=True):
            raw_speed = data_edge.get('maxspeed', 40)
            if isinstance(raw_speed, list): raw_speed = raw_speed[0]
            try:
                if isinstance(raw_speed, str): speed_kph = float(''.join(filter(str.isdigit, raw_speed)) or 40.0)
                else: speed_kph = float(raw_speed)
            except: speed_kph = 40.0 
                
            data_edge['speed_kph'] = speed_kph
            length_m = float(data_edge.get('length', 100.0))
            data_edge['base_travel_time'] = length_m / (data_edge['speed_kph'] * 1000 / 3600)
            
            highway_type = data_edge.get('highway', '')
            if isinstance(highway_type, list): highway_type = highway_type[0]
            major_roads = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'secondary']
            
            assigned_multiplier = model_multipliers[edge_index % num_sensors]
            if highway_type in major_roads:
                edge_multiplier = assigned_multiplier
            else:
                edge_multiplier = max(1.0, 1.0 + (assigned_multiplier - 1.0) * 0.15)
                
            data_edge['predicted_travel_time'] = data_edge['base_travel_time'] * edge_multiplier
            edge_index += 1

            if highway_type in major_roads:
                color = '#10b981' if edge_multiplier <= 1.5 else '#f59e0b' if edge_multiplier <= 2.8 else '#ef4444'
                if 'geometry' in data_edge: coords = [(lat, lon) for lon, lat in list(data_edge['geometry'].coords)]
                else: coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                heatmap_lines.append({'coords': coords, 'color': color})

        start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
        end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
        shortest_route = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        fastest_route = nx.shortest_path(G, source=start_node, target=end_node, weight='predicted_travel_time')

        # --- MAP GENERATION ---
        m = folium.Map(tiles='CartoDB dark_matter')
        
        for line in heatmap_lines:
            folium.PolyLine(locations=line['coords'], color=line['color'], weight=3, opacity=0.4).add_to(m)

        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in shortest_route], color='#3b82f6', weight=4, opacity=0.7, dash_array='10, 10').add_to(m)
        folium.PolyLine([[G.nodes[n]['y'], G.nodes[n]['x']] for n in fastest_route], color='#06b6d4', weight=6, opacity=1.0).add_to(m)
        
        folium.Marker(location=start_coords, icon=folium.Icon(color='green', icon='play')).add_to(m)
        folium.Marker(location=end_coords, icon=folium.Icon(color='red', icon='stop')).add_to(m)

        # Auto-frame perfectly around the route
        m.fit_bounds([start_coords, end_coords])
        
        # --- ONLY ONE LEGEND (TOP RIGHT) ---
        route_legend = '''
        <div style="position: absolute; top: 20px; right: 20px; width: 220px; background-color: rgba(9, 9, 11, 0.95); border: 1px solid rgba(59, 130, 246, 0.5); border-radius: 8px; z-index: 999999; font-size: 12px; color: white; padding: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.8); font-family: 'Inter', sans-serif;">
            <b style="font-size:13px; color:#60a5fa; margin-bottom:10px; display:block;">Route Legend</b>
            <div style="margin-bottom:8px; display:flex; align-items:center;">
                <i style="background:#06b6d4; width:16px; height:4px; border-radius:2px; margin-right:8px; box-shadow: 0 0 5px #06b6d4;"></i> Predictive (Fastest)
            </div>
            <div style="margin-bottom:12px; display:flex; align-items:center;">
                <i style="background:transparent; border-top: 3px dashed #3b82f6; width:16px; margin-right:8px;"></i> Baseline (Shortest)
            </div>
            <b style="font-size:11px; color:#a1a1aa; display:block; text-transform:uppercase; margin-bottom:6px; letter-spacing: 0.5px;">Predicted Traffic</b>
            <div style="margin-bottom:4px; display:flex; align-items:center;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Free Flow</div>
            <div style="margin-bottom:4px; display:flex; align-items:center;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Moderate</div>
            <div style="display:flex; align-items:center;"><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; margin-right:8px;"></i> Heavy Congestion</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(route_legend))
        
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)