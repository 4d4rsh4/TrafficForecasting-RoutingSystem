import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import folium
import random

# --- 1. SETUP FLASK APP ---
app = Flask(__name__)
CORS(app) # Allows your frontend to talk to this backend

# --- 2. LOAD YOUR PRE-TRAINED MODEL ---
# IMPORTANT: Copy the TrafficPredictor class definition from your script here.
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

# Load the model state
model = TrafficPredictor(input_dim=309, hidden_dim=64, num_layers=2, output_steps=12, num_sensors=307)
model.load_state_dict(torch.load('traffic_model_temporal.pth'))
model.eval() # Set model to evaluation mode

# --- 3. LOAD & PREPARE DATA FOR PREDICTION CONTEXT ---
# We need the scaler and historical data to make new predictions
raw_data = np.load('data/pems04.npz')['data']
data = raw_data[:, :, 0]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

time_index = pd.date_range("2018-01-01 00:00:00", periods=data.shape[0], freq='5min')
hours_scaled = time_index.hour.values.reshape(-1, 1) / 23.0
days_scaled = time_index.dayofweek.values.reshape(-1, 1) / 6.0
data_with_time = np.hstack([data_scaled, hours_scaled, days_scaled])

print("Model and data loaded successfully.")

# --- 4. THE MAIN API ENDPOINT ---
# --- 4. THE MAIN API ENDPOINT ---
@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        # --- A. Get user input from the frontend ---
        data_input = request.json
        start_coords = (float(data_input['start_lat']), float(data_input['start_lon']))
        end_coords = (float(data_input['end_lat']), float(data_input['end_lon']))
        
        target_day_str = data_input['day']
        target_hour = int(data_input['hour'])
        
        print(f"--- New Simulation Request: {target_day_str} at {target_hour}:00 ---")

        # --- B. Find the correct time window in the dataset ---
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        target_day_int = day_map[target_day_str]
        
        # Start searching in the test set portion (like in data_exploration.py)
        split_idx = int(0.8 * len(time_index)) 
        found_idx = -1
        
        for i in range(split_idx, len(time_index) - 12):
            if time_index[i].dayofweek == target_day_int and time_index[i].hour == target_hour:
                found_idx = i
                break
                
        if found_idx == -1:
            # Fallback just in case the exact hour isn't in the 20% test split
            print("Exact time not found in test split, using a fallback window.")
            found_idx = split_idx + 50 

        # Extract the 12-step window leading up to this hour
        context_window = data_with_time[found_idx - 12 : found_idx, :]
        window_tensor = torch.FloatTensor(context_window).unsqueeze(0)

        # --- C. Run the GCN-GRU Model Prediction ---
        with torch.no_grad():
            prediction_scaled = model(window_tensor)
            
        pred_t1_scaled = prediction_scaled[0, 0, :].numpy() 
        pred_t1_real_flow = scaler.inverse_transform(pred_t1_scaled.reshape(-1, 1)).flatten()
        
        # Create an array of 307 specific congestion multipliers
        model_multipliers = []
        for flow in pred_t1_real_flow:
            # Scale flow to a multiplier (1.0 = fast, up to 4.0 = slow)
            factor = max(1.0, 1.0 + (flow - 100) / 100.0) 
            factor = min(factor, 4.0) 
            model_multipliers.append(factor)

        # --- D. Load Road Network (I-94 Bounding Box) ---
        north = max(start_coords[0], end_coords[0]) + 0.015
        south = min(start_coords[0], end_coords[0]) - 0.015
        east = max(start_coords[1], end_coords[1]) + 0.015
        west = min(start_coords[1], end_coords[1]) - 0.015
        
        bbox = (west, south, east, north)
        G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
        
        # --- E. Apply Model Predictions & Build Heatmap ---
        heatmap_lines = [] 
        edge_index = 0
        num_sensors = len(model_multipliers)

        for u, v, key, data_edge in G.edges(keys=True, data=True):
            # Clean up speed limits
            raw_speed = data_edge.get('maxspeed', 40)
            if isinstance(raw_speed, list): raw_speed = raw_speed[0]
            try:
                if isinstance(raw_speed, str):
                    speed_val = ''.join(filter(str.isdigit, raw_speed))
                    speed_kph = float(speed_val) if speed_val else 40.0
                else: speed_kph = float(raw_speed)
            except Exception:
                speed_kph = 40.0 
                
            data_edge['speed_kph'] = speed_kph
            length_m = float(data_edge.get('length', 100.0))
            data_edge['base_travel_time'] = length_m / (data_edge['speed_kph'] * 1000 / 3600)
            
            # --- INTELLIGENT ROUTING LOGIC ---
            highway_type = data_edge.get('highway', '')
            if isinstance(highway_type, list): highway_type = highway_type[0]
            major_roads = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'secondary']
            
            # Grab a specific prediction from your model's 307 outputs
            assigned_multiplier = model_multipliers[edge_index % num_sensors]
            
            if highway_type in major_roads:
                edge_multiplier = assigned_multiplier
            else:
                # Local roads get much less congestion
                edge_multiplier = max(1.0, 1.0 + (assigned_multiplier - 1.0) * 0.15)
                
            data_edge['predicted_travel_time'] = data_edge['base_travel_time'] * edge_multiplier
            edge_index += 1

            # --- EXTRACT GEOMETRY FOR SMOOTH HEATMAP LINES ---
            if highway_type in major_roads:
                if edge_multiplier <= 1.5:
                    color = '#10b981' # Green (Clear)
                elif edge_multiplier <= 2.8:
                    color = '#f59e0b' # Orange (Moderate)
                else:
                    color = '#ef4444' # Red (Heavy Traffic)

                # Safely extract the road shape
                if 'geometry' in data_edge:
                    coords = [(lat, lon) for lon, lat in list(data_edge['geometry'].coords)]
                else:
                    coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                
                heatmap_lines.append({'coords': coords, 'color': color})

        # --- F. Calculate Routes ---
        start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
        end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
        
        shortest_route = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        fastest_route = nx.shortest_path(G, source=start_node, target=end_node, weight='predicted_travel_time')

        # --- G. Generate Folium Map ---
        map_center = ((start_coords[0] + end_coords[0])/2, (start_coords[1] + end_coords[1])/2)
        m = folium.Map(location=map_center, zoom_start=13, tiles='CartoDB dark_matter')
        
        # 1. DRAW HEATMAP (Underneath)
        for line in heatmap_lines:
            folium.PolyLine(locations=line['coords'], color=line['color'], weight=3, opacity=0.5).add_to(m)

        # 2. DRAW BASELINE ROUTE (Dashed Blue - Shortest Distance)
        folium.PolyLine(
            [[G.nodes[n]['y'], G.nodes[n]['x']] for n in shortest_route], 
            color='#3b82f6', weight=4, opacity=0.7, dash_array='10, 10', tooltip="Baseline: Shortest Distance"
        ).add_to(m)
        
        # 3. DRAW PREDICTIVE ROUTE (Solid Cyan - Smart Route)
        folium.PolyLine(
            [[G.nodes[n]['y'], G.nodes[n]['x']] for n in fastest_route], 
            color='#06b6d4', weight=6, opacity=1.0, tooltip=f"GCN-GRU Predictive Route ({target_day_str} {target_hour}:00)"
        ).add_to(m)
        
        # Add Markers
        folium.Marker(location=start_coords, popup="Start", icon=folium.Icon(color='green', icon='play')).add_to(m)
        folium.Marker(location=end_coords, popup="End", icon=folium.Icon(color='red', icon='stop')).add_to(m)

        # --- 4. ADD UI LEGEND ---
        legend_html = '''
        <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; height: auto; 
                    background-color: rgba(30, 41, 59, 0.9); border: 1px solid #3b82f6; border-radius: 10px;
                    z-index:9999; font-size:13px; color: white; padding: 15px; font-family: sans-serif;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <b style="font-size:15px; color:#60a5fa; margin-bottom:10px; display:block;">Map Legend</b>
            
            <div style="margin-bottom:8px;"><i style="background:#06b6d4; width:15px; height:4px; float:left; margin-right:10px; margin-top:5px;"></i> Predictive Route (Fastest)</div>
            <div style="margin-bottom:15px;"><i style="background:#3b82f6; width:15px; height:4px; float:left; margin-right:10px; margin-top:5px; border-top: 2px dashed #0f172a;"></i> Baseline Route (Shortest)</div>
            
            <b style="font-size:12px; color:#94a3b8; margin-bottom:5px; display:block; text-transform:uppercase;">Predicted Traffic</b>
            <div style="margin-bottom:4px;"><i style="background:#10b981; width:10px; height:10px; border-radius:50%; float:left; margin-right:10px; margin-top:2px;"></i> Free Flow</div>
            <div style="margin-bottom:4px;"><i style="background:#f59e0b; width:10px; height:10px; border-radius:50%; float:left; margin-right:10px; margin-top:2px;"></i> Moderate</div>
            <div style="margin-bottom:4px;"><i style="background:#ef4444; width:10px; height:10px; border-radius:50%; float:left; margin-right:10px; margin-top:2px;"></i> Heavy Congestion</div>
        </div>
        '''
        from folium import Element
        m.get_root().html.add_child(Element(legend_html))
        
        return jsonify({'map_html': m._repr_html_()})
    except Exception as e:
        import traceback
        traceback.print_exc() # Prints the full error in the terminal for debugging
        return jsonify({'error': str(e)}), 500
    
# To run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)