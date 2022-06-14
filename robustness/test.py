import numpy as np
import itertools
import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from Clustering_distance import Cluster_distance,Cluster
from scipy.optimize import linear_sum_assignment
features, true_labels = make_blobs(
   n_samples=200,
   centers=3,
   cluster_std=2.75,
   random_state=42)
print(np.array(features).shape)
[U,_,V]=np.linalg.svd(np.random.rand(np.array(features).shape[1],np.array(features).shape[1]))
roat=U.dot(V.T)
cluster1=Cluster(features, 3)
cluster2=Cluster(features.dot(roat), 3)
cluster_distance=Cluster_distance(cluster1,cluster2,3)
cost=cluster_distance.cost()
distance=cluster_distance.distance()
#cost = np.array([[0, 1, 3], [2, 0, 5], [3, 2, 0]])
print(cost)
print(distance)




