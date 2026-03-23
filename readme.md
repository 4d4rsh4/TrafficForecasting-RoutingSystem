🚦 TrafficControl: Spatio-Temporal Predictive Routing Engine
![alt text](https://img.shields.io/badge/Status-Active-success.svg)

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue.svg)

![alt text](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)

![alt text](https://img.shields.io/badge/Type-Major_Project_[CT707]-8A2BE2.svg)
TrafficControl is an advanced AI-driven web application designed to forecast urban traffic congestion and dynamically calculate mathematically optimal driving routes. Developed as a final-year Major Project [CT707], it shifts traffic management from a reactive state (responding to existing jams) to a proactive state (rerouting before gridlock occurs).
📖 Table of Contents
Overview
Key Features
System Architecture
Tech Stack
Project Structure
Installation & Setup
Usage Guide
Development Team
🌍 Overview
Traditional GPS routing algorithms calculate paths based on static distances or current, real-time traffic. This often leads to secondary congestion, as all vehicles are routed onto the same "fastest" path.
TrafficControl solves this by utilizing deep learning to look into the future. By forecasting the Spatio-Temporal traffic state of a road network hours in advance, the system dynamically alters the "weight" (travel time) of road segments. It then uses Dijkstra's algorithm to compute an AI-optimized route that bypasses impending congestion.
The system is trained on the benchmark PeMS04 (San Francisco Bay Area) dataset and features a novel Heuristic Topology Transfer logic, allowing the predictive models to be applied to different geographical regions (like Kathmandu, Nepal) without requiring massive localized sensor datasets.
✨ Key Features
Deep Learning Forecasting: Utilizes a custom GCN-GRU (Graph Convolutional Network + Gated Recurrent Unit) architecture to predict future traffic flows.
Dynamic Dijkstra Routing: Re-weights edges on real-world OSM (OpenStreetMap) road graphs based on AI predictions to find the true fastest path.
Spatio-Temporal Awareness: The model understands both spatial relationships (how intersections connect) and temporal trends (time-of-day/day-of-week traffic build-up).
Interactive Simulation UI: A sleek, dark-mode web interface allowing users to drop pins, set future dates/times, and visualize the AI's route vs. the standard shortest path.
Topology Transfer: Built-in heuristic logic to map high-highway traffic trends from California to complex, lower-capacity road networks like Kathmandu.
🧠 System Architecture
The core intelligence of TrafficControl relies on combining Graph Theory with Recurrent Neural Networks:
GCN (Graph Convolutional Network): Models the city as a mathematical graph. It captures the physical connectivity between intersections, understanding that a jam at point A will likely spill over to connected point B.
GRU (Gated Recurrent Unit): Processes historical time-series data to understand the flow of time, recognizing patterns like morning rush hours or weekend lulls.
OSMnx & NetworkX Engine: Downloads physical road networks, converts them into traversable graphs, and applies the AI outputs as "travel time penalties" on specific roads.
🛠 Tech Stack
AI & Data Science (Backend):
Python 3
PyTorch (Deep Learning Model)
Scikit-Learn (Data Scaling)
NumPy & Pandas (Data Manipulation)
Graph Processing & Routing:
OSMnx (OpenStreetMap data retrieval)
NetworkX (Graph algorithms & Dijkstra routing)
Web Application:
Flask (API Server)
Folium (Backend Python Map Rendering)
HTML5, CSS3, Vanilla JS (Frontend)
Leaflet.js (Frontend Interactive Maps)
📂 Project Structure
Ensure your repository is structured exactly like this before running:
code
Text
TrafficControl/
│
├── data/                           # Required dataset folder
│   ├── pems04.npz                  # Raw traffic flow sensor data
│   └── distance04.csv              # Distance matrix for GCN adjacency
│
├── traffic_model_temporal.pth      # Pre-trained PyTorch weights
├── app.py                          # Main Flask server & Routing Engine
├── setup_map.py                    # Pre-caches OpenStreetMap GraphML
├── index.html                      # Frontend: Home & System Overview
├── simulate.html                   # Frontend: Interactive Simulation UI
├── logo.png                        # Application Logo
├── script.js                       # Frontend logic
├── style.css                       # Global styles
└── README.md
🚀 Installation & Setup
Follow these steps to run the simulation locally on your machine.
1. Clone the repository
code
Bash
git clone https://github.com/yourusername/TrafficControl.git
cd TrafficControl
2. Create a Virtual Environment (Recommended)
code
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
code
Bash
pip install torch numpy pandas scikit-learn flask flask-cors osmnx networkx folium
4. Cache the Map Data (Important)
To prevent long loading times during demonstrations, run the setup script to pre-download the San Francisco road network:
code
Bash
python setup_map.py
5. Start the Server
code
Bash
python app.py
The Flask backend is now running on http://127.0.0.1:5000.
6. Launch the App
Simply open index.html in your preferred web browser (Chrome/Edge/Firefox), or use a Live Server extension.
💻 Usage Guide
Overview Page: Upon opening index.html, you will see a live global map rendering the current traffic state generated by the GCN-GRU model.
Launch Simulation: Click "Launch Simulation" and select your target city (San Francisco or Kathmandu).
Drop Pins: On the simulation page, click the map to set a Start Point (Green) and an End Point (Red).
Set Parameters: Choose a day of the week and a time of day (e.g., Friday, 17:00) to tell the AI when you plan to drive.
Run AI: Click "Run AI Simulation". The system will query the PyTorch model, alter the road graph weights, and display a map showing:
White Dashed Line: Standard Shortest Path.
Blue Solid Line: AI-Optimized Path (avoiding predicted traffic).
Stats Box: Total time saved compared to standard routing.
🎓 Development Team
This platform was researched and developed by final-year Computer Engineering students as a Major Project [CT707].
Adarsha Rai (KAT078BCT014) - Model Research & Data Pipeline
Agraj Singh Adhikari (KAT078BCT016) - System Integration & Backend
Amuhang Limbu Rai (KAT078BCT017) - Documentation & Analysis
Gaurav Adhikari (KAT078BCT034) - Frontend Architecture & UI/UX
Institution:
Kathmandu Engineering College (KEC), Kalimati
Tribhuvan University, Institute of Engineering (IOE), Nepal
Department of Computer Engineering