import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from joblib import dump, load

class ClusteringOPS:
    """
    Class with all clustering methods.
    """

    def __init__(self, clustering_method='kmeans', n_clusters=20, seed=0, **kwargs):
        
        self.available_methods = ['kmeans', 'gmm']
        self.clustering_method = clustering_method
        if clustering_method == 'kmeans':
            self.cluster = MiniBatchKMeans(n_clusters=n_clusters,
                                           batch_size=2048,
                                           n_init=5,
                                           random_state=seed,
                                           **kwargs)
            
        elif clustering_method == 'gmm':
            self.cluster = GaussianMixture(n_components=n_clusters,
                                           n_init=5,
                                           max_iter=10000,
                                           random_state=seed,
                                           reg_covar=1.25,
                                           **kwargs)
        else:
            raise NotImplementedError(f'Method {clustering_method} is not defined.' +
                                      f' Available options are: {self.available_methods}')
    
    def cluster_features(self, features):
        self.cluster.fit(features)
        
    def save_model(self, filename):
        dump(self.cluster, filename)
        
    def load_model(self, filename):
        self.cluster = load(filename)
    
    def predict_cluster(self, X):
        return self.cluster.predict(X)
    
    def predict_proba(self, X):
        if self.clustering_method == 'kmeans':
            return self.soft_assignment_kmeans(X)
        elif self.clustering_method == 'gmm':
            return self.cluster.predict_proba(X)
        else:
            raise NotImplementedError()
    
    def fit_predict(self, X):
        self.cluster_features(X)
        return self.predict_cluster(X)
    
    def soft_assignment_kmeans(self, X):
        distances = self.cluster.transform(X)
        distances += 1e-6 # Add small value in case some distance is 0 in order to do the inverse
        weights = 1 / distances # Invert distances to obtain larger weights for closer clusters
        probs = weights / weights.sum(axis=1, keepdims=True) # Normalize weights to add up to 1.
        return probs
        