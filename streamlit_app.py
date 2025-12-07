import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load artifacts
rfm_data = pd.read_csv("rfm_with_clusters.csv")
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("kmeans_model.pkl", "rb"))

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation & Prediction App")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to:", ["Dashboard Overview", "Predict Customer Segment"])

if page == "Dashboard Overview":
    st.subheader("📌 Clustered Dataset Preview")
    st.dataframe(rfm_data.head())

    st.subheader("📌 Cluster Distribution")
    st.bar_chart(rfm_data["Cluster"].value_counts())

    st.subheader("📌 Segment Insights")
    st.text("""
    Cluster 0 → Loyal / High value customers  
    Cluster 1 → Discount seekers / Medium spenders  
    Cluster 2 → New or inactive buyers  
    Cluster 3 → At-risk / churn segment
    """)

elif page == "Predict Customer Segment":
    st.subheader("🧮 Enter Customer RFM Values")

    recency = st.number_input("Recency (days since last purchase)", min_value=1, max_value=500)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100)
    monetary = st.number_input("Monetary (total spend)", min_value=1, max_value=100000)

    if st.button("Predict Segment"):
        new_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = model.predict(new_scaled)[0]

        st.success(f"Predicted Segment: Cluster {cluster}")

        explanations = {
            0: "Likely Loyal / High Value Customer",
            1: "Discount-Driven or Mid-Spend Customer",
            2: "New or Low Engagement Buyer",
            3: "At-Risk or Churn Customer"
        }

        st.write("📌 Explanation:", explanations.get(cluster, "Unknown Segment"))
