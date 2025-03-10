import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

def generate_datasets():
    # Dataset where DBSCAN performs well 
    X1, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # Dataset where DBSCAN struggle
    X2, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    
    return X1, X2

def apply_clustering(X, algorithm, params):
    clusterer = algorithm(**params)
    labels = clusterer.fit_predict(X)
    return labels

def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', alpha=0.75)
    plt.title(title)
    plt.show()

def main():
    # Generate datasets
    X1, X2 = generate_datasets()
    
    # Standardize the data 
    X1 = StandardScaler().fit_transform(X1)
    X2 = StandardScaler().fit_transform(X2)
    
    # Apply clustering algorithms to dataset 1 
    labels_kmeans_1 = apply_clustering(X1, KMeans, {'n_clusters': 2, 'random_state': 42})
    labels_hierarchical_1 = apply_clustering(X1, AgglomerativeClustering, {'n_clusters': 2})
    labels_dbscan_1 = apply_clustering(X1, DBSCAN, {'eps': 0.3, 'min_samples': 5})
    
    # Apply clustering algorithms to dataset 2 
    labels_kmeans_2 = apply_clustering(X2, KMeans, {'n_clusters': 3, 'random_state': 42})
    labels_hierarchical_2 = apply_clustering(X2, AgglomerativeClustering, {'n_clusters': 3})
    labels_dbscan_2 = apply_clustering(X2, DBSCAN, {'eps': 0.5, 'min_samples': 5})
    
    # Plot results
    datasets = [(X1, labels_kmeans_1, 'K-Means on Moons'),
                (X1, labels_hierarchical_1, 'Hierarchical Clustering on Moons'),
                (X1, labels_dbscan_1, 'DBSCAN on Moons'),
                (X2, labels_kmeans_2, 'K-Means on Blobs'),
                (X2, labels_hierarchical_2, 'Hierarchical Clustering on Blobs'),
                (X2, labels_dbscan_2, 'DBSCAN on Blobs')]
    
    for X, labels, title in datasets:
        plot_clusters(X, labels, title)

if __name__ == "__main__":
    main()


