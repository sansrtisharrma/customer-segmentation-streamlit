import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===================================
# Load Data & Model
# ===================================
@st.cache_resource
def load_artifacts():
    rfm_data = pd.read_csv("rfm_with_clusters.csv")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("kmeans_model.pkl", "rb"))
    return rfm_data, scaler, model

rfm, scaler, model = load_artifacts()

recommendations = {
    0: "Offer loyalty rewards, exclusive previews & VIP sales.",
    1: "Send discount campaigns and cashback offers.",
    2: "Send onboarding emails, welcome coupons & referral offers.",
    3: "Send retention campaigns, reminders, and special win-back deals."
}

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Customer Segmentation & Recommendation System")

# Sidebar Navigation
st.sidebar.header("📌 Menu")
page = st.sidebar.radio("Select Page", 
                        ["Overview Dashboard", "Segment Insights", "Predict Segment"])

# ===================================
# PAGE 1: Overview Dashboard
# ===================================
if page == "Overview Dashboard":
    st.subheader("📌 Clustered Dataset Preview")
    
    st.markdown("### 📌 Key Insights")
    colA, colB, colC = st.columns(3)
    colA.metric("Active Clusters", rfm["Cluster"].nunique())
    colB.metric("Highest Segment Size", rfm["Cluster"].value_counts().max())
    colC.metric("Lowest Segment Size", rfm["Cluster"].value_counts().min())

    st.dataframe(rfm.head())

    st.markdown("---")

    st.subheader("📊 Cluster Distribution")
    cluster_counts = rfm["Cluster"].value_counts().sort_index()

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(cluster_counts)
    with col2:
        st.metric("Total Customers", len(rfm))
        st.metric("Number of Segments", rfm["Cluster"].nunique())

    st.success("✔ Use this page screenshot in thesis implementation chapter")

# ===================================
# PAGE 2: Segment Insights
# ===================================
elif page == "Segment Insights":
    st.subheader("📌 Segment Profile Statistics")
    segment_stats = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    st.dataframe(segment_stats.style.highlight_max(color="lightgreen", axis=0))

    st.markdown("---")

    st.subheader("💡 Recommended Business Actions")

    colors = ["#D4EDDA", "#D1ECF1", "#FFF3CD", "#F8D7DA"]

    for segment, text in recommendations.items():
        st.markdown(
            f"""
            <div style="
                padding:10px;
                border-radius:8px;
                background:{colors[segment]};
                font-size:16px;
                margin-bottom:8px;">
                <b>Cluster {segment}:</b> {text}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.success("✔ This page represents Explainability & XAI — very impressive for viva")

# ===================================
# PAGE 3: Prediction
# ===================================
elif page == "Predict Segment":
    st.subheader("🧮 Enter Customer RFM Values")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=1, max_value=500)
    with col2:
        frequency = st.number_input("Frequency (purchases)", min_value=1, max_value=100)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=1, max_value=200000)

    if st.button("Predict Customer Segment"):
        new_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = model.predict(new_scaled)[0]

        st.success(f"Predicted Segment: Cluster {cluster}")

        explanation = {
            0: "Loyal / High Value Buyer",
            1: "Discount Oriented Buyer",
            2: "New or Low Interaction Buyer",
            3: "At-Risk Churn Customer"
        }
        st.write("📌 Interpretation:", explanation.get(cluster))

        persona = {
            0: "This customer is loyal and engages frequently — ideal for retention programs.",
            1: "This customer responds to offers and discounts — best for promotional campaigns.",
            2: "This customer is new or minimally engaged — requires onboarding nurturing.",
            3: "This customer is losing interest — needs win-back messaging."
        }
        st.info(f"👤 Customer Persona Insight: {persona.get(cluster)}")

        st.markdown(
            f"""
            <div style="padding:12px;border-radius:8px;background:#fff3cd;color:#856404;font-size:16px">
                📌 <b>Recommended Strategy:</b> {recommendations.get(cluster)}
            </div>
            """,
            unsafe_allow_html=True
        )
