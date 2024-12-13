import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = 'C:/Users/moham/Documents/Uni/Data Mining/cleaned_data.csv'
data = pd.read_csv(file_path)

# Select numerical columns for clustering
numerical_columns = [
    'Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 
    'Subject_6', 'Subject_7', 'Subject_8', 'Subject_9', 'Subject_10', 
    'AVG Score'
]
data_numerical = data[numerical_columns]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

# Apply K-Means clustering with a predefined number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_scaled)

# Get the cluster labels and centroids
labels = kmeans.predict(data_scaled)
centroids = kmeans.cluster_centers_

# Visualize the clustering using the first two features
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=50, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()
