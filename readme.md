# 🚦 Spatio-Temporal Traffic Forecasting & Predictive Routing [CT707]

This repository contains the source code for the Major Project **"Spatio-Temporal Traffic Forecasting and Congestion Mapping using GCN-GRU,"** submitted to the Department of Computer Engineering at Kathmandu Engineering College, Tribhuvan University.

The system utilizes a hybrid Deep Learning architecture (**GCN-GRU**) trained on the **PeMS-04 dataset** to predict future traffic flow. These predictions are fed into a dynamic routing engine that calculates optimal, time-efficient routes by anticipating and avoiding predicted congestion.

The entire system is served via a **Flask backend** to a modern, interactive web dashboard.

---

## 🌟 Features

* **AI-Powered Traffic Forecasting:**
  A `TemporalGCNGRU` model built with PyTorch captures both spatial (road layout) and temporal (time-based) traffic patterns.

* **Dynamic Dijkstra Routing:**
  Computes:

  * Shortest physical route (baseline)
  * "AI Optimized" route using predicted future traffic

* **Heuristic Topology Transfer:**
  Applies California-trained model behavior to other cities (demonstrated with Kathmandu).

* **Interactive Web Dashboard:**
  Modern "Glassmorphic" UI using HTML, CSS, and JavaScript.

* **Live Heatmap Visualization:**
  Traffic intensity visualization:

  * 🟢 Green → Free Flow
  * 🟠 Orange → Moderate
  * 🔴 Red → Heavy

* **GPU Acceleration:**
  Supports CUDA-enabled GPUs for fast training and inference (~sub-2 seconds routing).

---

## 🚀 Getting Started

### 📌 Prerequisites

* Python 3.9+
* (Recommended) NVIDIA GPU with CUDA installed

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd TrafficControl
```

### 2️⃣ Create & Activate Virtual Environment

```bash
# Create environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

#### 🟢 Option A: GPU Users (CUDA 12.1+)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn flask flask-cors osmnx networkx folium matplotlib
```

#### ⚪ Option B: CPU Users

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn flask flask-cors osmnx networkx folium matplotlib
```

---

## 💻 Running the Application

### ▶️ Start Backend Server

```bash
python app.py
```

> ⚠️ First run may take 1–2 minutes to download OpenStreetMap data.
> Wait until the terminal shows: **"Server is ready"**

---

### 🌐 Launch Frontend

* Go to the `frontend` folder
* Open `index.html` in your browser

---

## 🧪 Using the Simulation

1. Click **"Launch Simulation"**
2. Select deployment city (San Francisco / Kathmandu)
3. Choose:

   * Start point (Green)
   * End point (Red)
4. Select:

   * Day of Week
   * Time (0–23)
5. Click **"Run AI Simulation"**

➡️ View:

* Predicted traffic heatmap
* AI-optimized route vs shortest path

---

## 🧠 Training the Model (Optional)

To retrain the model:

```bash
python train_temporal_gcn_gru.py
```

* Runs for **50 epochs**
* Splits data: **80% train / 20% validation**
* Outputs **MSE loss**
* Saves model as: `traffic_model_temporal.pth`

To plot learning curves:

```bash
python plot_results.py
```

---

## 👥 Development Team

Developed as Major Project **[CT707]** by:

* **Adarsha Rai (KAT078BCT014)** — Model Research & Data Pipeline
* **Agraj Singh Adhikari (KAT078BCT016)** — System Integration & Backend
* **Amuhang Limbu Rai (KAT078BCT017)** — Documentation & Analysis
* **Gaurav Adhikari (KAT078BCT034)** — Frontend Architecture & UI/UX

📍 Kathmandu Engineering College
Department of Computer Engineering

---

## 📌 Notes

* Ensure GPU drivers and CUDA are correctly installed for acceleration.
* Internet connection required for initial map data download.
* Model performance depends on dataset quality and preprocessing.

---

## 📄 License

This project is developed for academic purposes. Licensing terms can be added as needed.
