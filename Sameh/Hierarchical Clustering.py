import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

file_path = 'C:/Users/moham/Documents/Uni/Data Mining/Final_dataset.csv'
data = pd.read_csv(file_path)

numerical_columns = [
    'Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 
    'Subject_6', 'Subject_7', 'Subject_8', 'Subject_9', 'Subject_10'
]
data_numerical = data[numerical_columns]

# Sample a subset of the data for clustering
sampled_data = data_numerical.sample(n=2000, random_state=42)

# Standardize the sampled data
scaler = StandardScaler()
sampled_scaled = scaler.fit_transform(sampled_data)

# Define linkage methods
linkage_methods = ['single', 'complete', 'average']

# Plot dendrograms for each linkage method one after another
for method in linkage_methods:
    # Perform hierarchical clustering
    Z = linkage(sampled_scaled, method=method)

    # Print the linkage matrix
    print(f"Linkage Matrix for {method.capitalize()} Linkage:")
    print(Z)
    print("\n" + "="*50 + "\n")

    # Plot the dendrogram
    plt.figure(figsize=(15, 7))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title(f'Hierarchical Clustering Dendrogram (AGNES) - {method.capitalize()} Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
