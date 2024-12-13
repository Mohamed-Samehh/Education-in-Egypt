import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

file_path = 'C:/Users/moham/Documents/Uni/Data Mining/cleaned_data.csv'
data = pd.read_csv(file_path)

# Select numerical columns for clustering
numerical_columns = [
    'Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 
    'Subject_6', 'Subject_7', 'Subject_8', 'Subject_9', 'Subject_10', 
    'AVG Score'
]
data_numerical = data[numerical_columns]

# Sample a subset of the data for clustering
sampled_data = data_numerical.sample(n=2000, random_state=42)

# Standardize the sampled data
scaler = StandardScaler()
sampled_scaled = scaler.fit_transform(sampled_data)

# Perform hierarchical clustering on the sampled data using 'single' linkage
Z_sampled_single = linkage(sampled_scaled, method='single')

# Plot the dendrogram for single linkage
plt.figure(figsize=(15, 7))
dendrogram(Z_sampled_single, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (AGNES) - Single Linkage")
plt.xlabel("Sample Index or Cluster Size")
plt.ylabel("Distance")
plt.show()

# Perform hierarchical clustering on the sampled data using 'complete' linkage
Z_sampled_complete = linkage(sampled_scaled, method='complete')

# Plot the dendrogram for complete linkage
plt.figure(figsize=(15, 7))
dendrogram(Z_sampled_complete, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (AGNES) - Complete Linkage")
plt.xlabel("Sample Index or Cluster Size")
plt.ylabel("Distance")
plt.show()

# Perform hierarchical clustering on the sampled data using 'average' linkage
Z_sampled_average = linkage(sampled_scaled, method='average')

# Plot the dendrogram for average linkage
plt.figure(figsize=(15, 7))
dendrogram(Z_sampled_average, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (AGNES) - Average Linkage")
plt.xlabel("Sample Index or Cluster Size")
plt.ylabel("Distance")
plt.show()
