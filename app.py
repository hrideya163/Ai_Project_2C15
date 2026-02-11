import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Building Energy Analytics", layout="wide")

required_files = [
    "artifacts/pca_scores_2d.csv",
    "artifacts/cluster_profiles.csv",
    "artifacts/building_segments.csv",
    "artifacts/k_sweep.csv",
    "artifacts/pca_explained_variance.csv"
]

for f in required_files:
    if not os.path.exists(f):
        st.error(f"Missing file: {f}. Please run train_model.py first.")
        st.stop()

st.title("🏢 Smart Building Energy Analytics Platform")
st.markdown("Unified platform for building-level energy analytics using PCA + KMeans clustering.")

@st.cache_data
def load_data():
    scores = pd.read_csv("artifacts/pca_scores_2d.csv")
    profiles = pd.read_csv("artifacts/cluster_profiles.csv")
    segments = pd.read_csv("artifacts/building_segments.csv")
    k_sweep = pd.read_csv("artifacts/k_sweep.csv")
    pca_ev = pd.read_csv("artifacts/pca_explained_variance.csv")
    return scores, profiles, segments, k_sweep, pca_ev

scores, profiles, segments, k_sweep, pca_ev = load_data()

st.sidebar.header("Building Selector")
building_ids = segments["building_id"].astype(str).unique().tolist()
selected_building = st.sidebar.selectbox("Select Building ID", building_ids)

b_row = segments[segments["building_id"].astype(str) == selected_building].iloc[0]
b_cluster = int(b_row["cluster"])

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visual Analytics",
    "🤖 Recommendations",
    "💡 Insights",
    "📈 Model Diagnostics"
])

# ---------- TAB 1: VISUAL ANALYTICS ----------
with tab1:
    st.subheader("📊 Visual Analytics")

    col_left, col_right = st.columns(2, gap="large")

    # ---- Left Column: PCA Plot + Same-Cluster Buildings ----
with col_left:
    st.markdown("### PCA-Based Building Positioning")
    st.caption("""
This scatter plot projects all buildings into a two-dimensional latent space derived from principal 
component analysis of multi-dimensional energy features. The relative proximity between points reflects 
similarity in consumption behavior. Highlighting the selected building enables rapid identification of 
peer buildings and anomaly detection within the broader energy usage landscape.
""")

    fig1, ax1 = plt.subplots(figsize=(4.8, 3.8))
    for c in sorted(scores["cluster"].unique()):
        dfc = scores[scores["cluster"] == c]
        ax1.scatter(dfc["pc1"], dfc["pc2"], alpha=0.4)

    b_point = scores[scores["building_id"].astype(str) == selected_building]
    ax1.scatter(b_point["pc1"], b_point["pc2"], s=110, marker="X", edgecolor="black")

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    st.pyplot(fig1, use_container_width=True)

    # ---- Same Cluster Buildings Section ----
    st.markdown("#### 🏷️ Peer Buildings in the Same Cluster")
    st.caption("""
This table lists buildings that belong to the same cluster as the selected building, indicating peers 
with similar energy consumption patterns and operational characteristics. Such peer grouping enables 
benchmarking, identification of best practices, and transfer of efficiency strategies across comparable 
buildings within the portfolio.
""")

    peer_buildings = segments[segments["cluster"] == b_cluster]["building_id"].astype(str).tolist()

    peer_df = pd.DataFrame({
        "Building ID": peer_buildings
    })

    st.dataframe(peer_df, use_container_width=True, height=180)


    # ---- Right Column: Energy Profile Bar Chart ----
    with col_right:
        st.markdown("### Building Energy Profile")
        st.caption("""
This bar chart summarizes key operational energy indicators for the selected building, including average 
load, peak demand, base load, and relative peak-to-off-peak and weekend-to-weekday ratios. These metrics 
jointly characterize consumption intensity, demand variability, and temporal imbalance, supporting rapid 
assessment of efficiency gaps and operational stress points.
""")

        features = ["avg_kw", "peak_kw", "base_kw", "peak_to_offpeak_ratio", "weekend_weekday_ratio"]

        fig2, ax2 = plt.subplots(figsize=(4.8, 3.8))
        ax2.bar(range(len(features)), b_row[features].values)
        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels(features, rotation=35, ha="right")
        ax2.set_ylabel("Energy Metric")
        st.pyplot(fig2, use_container_width=True)


# ---------- TAB 2: RECOMMENDATIONS ----------
with tab2:
    st.subheader("Automated Energy Optimization Recommendations")
    st.markdown("""
The following system-generated recommendations are derived from the building’s energy usage profile, 
appliance composition, and temporal demand patterns. These actions are intended to reduce operational 
costs, flatten peak demand, improve equipment efficiency, and enhance overall sustainability outcomes.
""")

    recs = []

    if b_row["hvac_share"] > 0.4:
        recs.append("• Optimize HVAC schedules using occupancy-aware controls and predictive temperature setpoints to reduce unnecessary cooling or heating loads.")
    if b_row["standby_share"] > 0.2:
        recs.append("• Deploy smart power strips and enforce device sleep policies to minimize persistent standby consumption during non-operational hours.")
    if b_row["shiftable_share"] > 0.3:
        recs.append("• Shift high-energy flexible loads such as EV charging, water heating, and laundry operations to off-peak tariff windows.")
    if b_row["peak_to_offpeak_ratio"] > 2.0:
        recs.append("• Implement peak shaving strategies, including battery storage dispatch or temporary load shedding during critical demand windows.")
    if b_row["weekend_weekday_ratio"] > 1.2:
        recs.append("• Audit weekend operational schedules to identify non-essential equipment that can be powered down outside business hours.")

    recs.extend([
        "• Introduce sub-metering to improve appliance-level visibility and enable finer-grained control strategies.",
        "• Perform periodic energy audits to detect equipment inefficiencies and aging infrastructure.",
        "• Leverage demand response programs offered by utilities to monetize flexible load potential.",
        "• Train occupants on energy-aware behavior to complement automated efficiency measures."
    ])

    for r in recs:
        st.success(r)

# ---------- TAB 3: INSIGHTS ----------
with tab3:
    st.subheader("Operational and Strategic Insights")
    st.markdown("""
The insights below interpret the building’s energy signature in operational terms and outline its strategic 
implications for facility management, energy planning, and sustainability governance.
""")

    st.markdown(f"""
- **Peer Benchmarking:** The building is grouped into Cluster {b_cluster}, indicating strong similarity with a cohort of buildings sharing comparable load profiles and appliance usage patterns.  
- **Demand Stability:** With a peak demand of {round(b_row['peak_kw'],2)} kW and a load factor of {round(b_row['load_factor'],2)}, the building exhibits {"stable" if b_row['load_factor'] > 0.6 else "highly variable"} demand characteristics.  
- **Primary Load Driver:** HVAC accounts for approximately {round(b_row['hvac_share']*100,1)}% of total electricity usage, positioning thermal management as the dominant optimization lever.  
- **Flexibility Potential:** Nearly {round(b_row['shiftable_share']*100,1)}% of the load is temporally flexible, offering substantial scope for time-of-use optimization and demand response participation.  
- **Operational Risk:** Elevated peak-to-off-peak ratios indicate potential exposure to tariff penalties and grid stress during high-demand intervals.  
- **Sustainability Implication:** Targeted efficiency improvements in HVAC and base load can yield disproportionate reductions in carbon intensity and operating expenditure.
""")

    st.info("""
Collectively, these insights support prioritization of retrofitting investments, scheduling reforms, and 
participation in grid-interactive efficient building (GEB) programs.
""")

# ---------- TAB 4: MODEL DIAGNOSTICS ----------
with tab4:
    st.subheader("Clustering & PCA Diagnostics")
    st.caption("""
These diagnostics summarize the quality of the unsupervised learning pipeline. The silhouette curve 
indicates how well buildings are separated into coherent clusters, while the PCA explained variance 
curve shows how effectively high-dimensional energy features are compressed into a low-dimensional 
representation without excessive information loss.
""")

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(k_sweep["k"], k_sweep["silhouette"], marker="o")
    ax3.set_xlabel("Number of Clusters (K)")
    ax3.set_ylabel("Silhouette Score")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.plot(pca_ev["component"], pca_ev["cumulative_explained_variance"], marker="o")
    ax4.set_xlabel("PCA Components")
    ax4.set_ylabel("Cumulative Explained Variance")
    st.pyplot(fig4)

st.markdown("---")
st.subheader("🌍 Real-World Applicability")
st.markdown("""
This platform demonstrates how machine learning can be operationalized for building energy management, 
enabling data-driven decision-making for cost reduction, grid responsiveness, and sustainability planning 
across campuses, enterprises, and smart city deployments.
""")
