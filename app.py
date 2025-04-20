# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:41:03 2025

@author: LAB
"""
import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
# Set the page config
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Set title
st.title("🔍 K-Means Clustering Visualizer")

# Display cluster centers
st.subheader("📊 Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# Load from a save dataset or generate synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Plot the result
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='o', label='Centroids')
ax.legend()

# Show the plot
st.pyplot(fig)
