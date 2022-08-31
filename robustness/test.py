import numpy as np
#import sys
from utils import load_data,load_npz
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch

f=open("hhh.txt", "w")
f.write("test")
# K=2*np.log(99)
# filepath="./GraphData/datasets/cora.npz"
# adj,edge,feature,_,node_label_onehot,_=load_npz(filepath)
# print(node_label_onehot.shape)
# adj_ground=node_label_onehot.dot(node_label_onehot.T)
# # adj=adj.to_dense()
# M=K*adj_ground-K/2
# # M=torch.clamp(torch.log(torch.div(adj,torch.ones(adj_ground.shape)-adj)), min=-1000,max=1000)
# L=np.linalg.eigvalsh(adj_ground)


# print(L[-100:-1])
#print(torch.sum(torch.gt(L,torch.zeros(2708))))
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from Clustering_distance import Cluster_distance,Cluster
from scipy.optimize import linear_sum_assignment
# import matplotlib.pyplot as plt
#plt.plot([1, 2, 3, 4])
# features, true_labels = make_blobs(
#    n_samples=200,
#    centers=5,
#    cluster_std=2.75,
#    random_state=42)
# print(np.array(features).shape)
# [U,_,V]=np.linalg.svd(np.random.rand(np.array(features).shape[1],np.array(features).shape[1]))
# roat=U.dot(V.T)
# cluster1=Cluster(features, 3)
# cluster2=Cluster(features.dot(roat), 3)
# label1=cluster1.clustering()
# label2=cluster1.clustering()
# print(label1,label2)
# print(normalized_mutual_info_score(label1, label2))
# cluster_distance=Cluster_distance(cluster1,cluster2,3)
# cost=cluster_distance.cost()
# distance=cluster_distance.distance()
# #cost = np.array([[0, 1, 3], [2, 0, 5], [3, 2, 0]])
# print(cost)
# print(distance)
# datetimeInstance=datetime.datetime.today()
# f = open(str(datetimeInstance)+".txt", "w")
# for item in f:
#     f.write("")

import torch.nn as nn
import torch
from Clustering_distance import Cluster_distance
from model import  GCN_naive
#
#
# for i in range(5):
#     np.random.seed(1)
#     torch.manual_seed(1)
#     torch.cuda.manual_seed(1)
#     print(torch.randn(3, 1))
#     model= GCN_naive(10, 10, 2,10)
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(name,param)




