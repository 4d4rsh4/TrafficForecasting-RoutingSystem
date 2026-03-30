🚦 Spatio-Temporal Traffic Forecasting & Predictive Routing (CT707)

This repository presents the implementation of the major project:

“Spatio-Temporal Traffic Forecasting and Congestion Mapping using GCN-GRU”
submitted to the Department of Computer Engineering, Kathmandu Engineering College, Tribhuvan University.

The system integrates deep learning-based traffic prediction with a dynamic routing engine to generate time-efficient navigation paths by anticipating future congestion.

A hybrid Graph Convolutional Network – Gated Recurrent Unit (GCN-GRU) model is trained on the PeMS-04 dataset, and its predictions are used to dynamically optimize routing decisions in real time.

The complete system is deployed using a Flask backend with an interactive web-based dashboard.

🌟 Key Features
🧠 AI-Based Traffic Forecasting

A custom TemporalGCNGRU model implemented in PyTorch captures:

Spatial dependencies (road network structure)
Temporal patterns (time-of-day and day-of-week variations)
🛣️ Intelligent Routing Engine

Routing is performed using Dijkstra’s Algorithm enhanced with the BPR delay function, enabling realistic travel-time estimation:

Shortest Path (baseline) — distance-based routing
AI-Optimized Path — avoids predicted congestion zones
🚑 Emergency Isochrone Analysis
Implements Radial Dijkstra Search (nx.ego_graph)
Generates reachability zones based on time constraints
Simulates emergency response coverage under predicted traffic conditions
🚧 Interactive Roadblock Simulation
Users can dynamically introduce disruptions such as:
Accidents
Road closures
Flooding or VIP movement
The system updates graph topology in real time and recomputes optimal routes
🌍 Heuristic Topology Transfer
Transfers learned temporal traffic behavior from California (PeMS-04)
Applies it to other city networks (e.g., Kathmandu)
Enables deployment in data-scarce environments
🗺️ Live Traffic Heatmap

Visualized using Folium:

Green → Free Flow
Orange → Moderate Traffic
Red → Heavy Congestion
⚡ GPU Acceleration
Supports CUDA-enabled GPUs
Enables fast inference and near real-time routing (sub-2 seconds)
🚀 Getting Started
📌 Prerequisites
Python 3.9+
(Recommended) NVIDIA GPU with CUDA support
⚙️ Setup & Installation
1. Clone the Repository
git clone <your-repository-url>
cd TrafficControl
2. Create & Activate Virtual Environment
# Create environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
3. Install Dependencies
GPU Users (CUDA 12.1+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn flask flask-cors osmnx networkx folium matplotlib
CPU Users
pip install torch torchvision torchaudio numpy pandas scikit-learn flask flask-cors osmnx networkx folium matplotlib
4. Preload Map Data (Recommended)
python setup_map.py

This caches road network data locally to prevent browser timeouts and improve loading speed.

💻 Running the Application
Start Backend Server
python app.py

Wait until:

✅ Advanced Model Loaded.

Server runs at:
http://127.0.0.1:5000

Launch Frontend
Navigate to the frontend folder
Open index.html in your browser
🧪 Using the Dashboard
A. Routing Simulation
Click "Launch Simulation"
Select city
Left-click to set Start and End points
Right-click to add roadblocks
Select day and time
Click "Run AI Simulation"

Outputs:

Traffic heatmap
Shortest vs AI-optimized route comparison
B. Emergency Isochrones
Open "Emergency Isochrones" tab
Select location or hospital
Set time limit and prediction time
Click "Generate Isochrone Map"

Output:

Reachable area under predicted traffic
🧠 Model Training (Optional)
Train Model
python train_temporal_gcn_gru.py
50 epochs
80/20 train-validation split
Output: traffic_model_temporal.pth
Plot Results
python plot_results.py

Generates loss curves and convergence plots.

👥 Development Team
Adarsha Rai — Model Research & Data Pipeline
Agraj Singh Adhikari — System Integration & Backend
Amuhang Limbu Rai — Documentation & Analysis
Gaurav Adhikari — Frontend Architecture & UI/UX

Kathmandu Engineering College
Department of Computer Engineering

📌 Notes
Ensure CUDA and GPU drivers are properly configured
Internet required only for initial map download
Model performance depends on data quality and preprocessing