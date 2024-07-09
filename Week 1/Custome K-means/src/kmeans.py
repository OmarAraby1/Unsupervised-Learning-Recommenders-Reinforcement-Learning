import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=0.1, random_state=None):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter      # Maximum number of iterations
        self.tol = tol                # Tolerance for convergence (minimum decrease in loss)
        self.random_state = random_state  # Random state for reproducibility
        self.centroids = None         # Centroids of clusters
        self.labels = None            # Labels of data points (cluster assignments)
        self.inertia = None           # Final value of the loss function (sum of squared distances)

    def fit(self, X):
        # Initialize centroids randomly from the data
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        
        # Iterate until convergence or max_iter is reached
        for _ in range(self.max_iter):
            # Assign labels based on closest centroid
            distances = cdist(X, self.centroids, metric='euclidean')
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids based on mean of points in each cluster
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Calculate the change in centroids
            centroid_change = np.sum(np.sqrt(np.sum((new_centroids - self.centroids) ** 2, axis=1)))
            
            # Update centroids
            self.centroids = new_centroids
            
            # Calculate the inertia (sum of squared distances)
            self.inertia = np.sum(np.min(distances, axis=1))
            
            # Check convergence
            if centroid_change < self.tol:
                break
    
    def predict(self, X):
        # Predict the closest cluster each sample in X belongs to
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
