# Smart Building Energy Analytics Platform

This project implements an end-to-end machine learning pipeline for analyzing smart building energy
consumption. The system generates building-level energy features, applies PCA for dimensionality
reduction, clusters buildings using K-Means, and presents results through an interactive Streamlit
dashboard with automated recommendations and actionable insights.

---

## 📁 Project Structure


---

## 🧩 File Descriptions

### 1️⃣ `generate.ipynb` — Data Generation Script  
Generates or prepares synthetic smart meter time-series data for multiple buildings, including
appliance-level energy consumption. This enables development and testing of the analytics pipeline
without relying on proprietary or sensitive real-world datasets.

**Purpose:**  
- Create reproducible input data  
- Simulate realistic building energy usage patterns  

---

### 2️⃣ `main.ipynb` — Main Training & Feature Engineering Script  
Extracts building-level energy features (average load, peak demand, load factor, variability, temporal
ratios, and appliance shares), scales the features, applies PCA for dimensionality reduction, and
clusters buildings using K-Means. All outputs and trained models are saved as artifacts for downstream
visualization.

**Purpose:**  
- Build energy features from raw meter data  
- Train PCA + K-Means models  
- Export artifacts for analysis and visualization  

---

### 3️⃣ `app.py` — Streamlit Interactive Dashboard  
Provides an interactive dashboard to explore building-level analytics. Users can visualize buildings
in PCA space, inspect energy profiles, view peer buildings in the same cluster, and receive automated
recommendations and operational insights.

**Purpose:**  
- Enable interactive visual analytics  
- Support building-level drill-down and peer benchmarking  
- Demonstrate real-world applicability of ML-driven energy management  

---

## ▶️ How to Run

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
### Step 2: Generate Data
```bash
python data_generation.py

