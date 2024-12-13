import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = 'C:/Users/moham/Documents/Uni/Data Mining/Final_dataset.csv'
data = pd.read_csv(file_path)

numerical_columns = [
    'Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 
    'Subject_6', 'Subject_7', 'Subject_8', 'Subject_9', 'Subject_10'
]
data_numerical = data[numerical_columns]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Apply K-Means clustering with a predefined number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_pca)

# Get the cluster labels and centroids
labels = kmeans.predict(data_pca)
centroids = kmeans.cluster_centers_

# Visualize the clustering in 2D (using PCA-reduced data)
plt.figure(figsize=(10, 6))
for cluster in range(4):
    cluster_points = data_pca[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
