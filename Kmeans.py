import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'D:/BUE- ICS/Fourth year/Semester one/Data Mining/Project/ahh ah/destination_export.csv'
data = pd.read_csv(data_path)

# Select features for clustering (example: Subject scores)
X = data[['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4']]  # Add more if needed

# Normalize the data to ensure all features contribute equally
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Step 1: Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_normalized)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Step 2: Apply K-Means with the optimal number of clusters
optimal_k = 3  # Replace this with the k determined from the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_normalized)

# Step 3: Add cluster labels to the dataset
data['Cluster'] = clusters

# Visualize the clusters (for 2D data)
plt.figure(figsize=(8, 5))
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (e.g., Subject_1)')
plt.ylabel('Feature 2 (e.g., Subject_2)')
plt.show()

# Step 4: Analyze cluster centroids
centroids = kmeans.cluster_centers_
print("Cluster Centroids (Normalized):", centroids)
