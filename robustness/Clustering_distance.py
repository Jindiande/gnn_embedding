#  [1] Von Luxburg, Ulrike. "Clustering stability: an overview." (2010).


import numpy as np
from sklearn.cluster import KMeans
from Method import center_index_node,non_overlap_num_one_axis
import itertools
from scipy.optimize import linear_sum_assignment

class Cluster_distance():# using linear assignment for implementing equation (2.2) in [1]
    def __init__(self,label1, label2, clusters_num):
        self.label1=label1
        self.label2=label2
        self.clusters_num = np.intc(clusters_num)

    def cost(self):# cost matrix for linear assginment problem
        center_dic_1=center_index_node(self.label1,self.clusters_num)
        center_dic_2=center_index_node(self.label2,self.clusters_num)
        center_dic_1=np.tile(np.expand_dims(center_dic_1,axis=2),(1,1,self.clusters_num))# k*m1*k
        center_dic_2 = np.tile(np.expand_dims(center_dic_2, axis=2), (1,1, self.clusters_num))#k*m2*k
        center_dic_2=np.transpose(center_dic_2,(2,1,0))# transpsose for dimen 0 and dimen 2
        cost=non_overlap_num_one_axis(center_dic_1, center_dic_2, 1)# return k*k cost matrix
        return cost
    def distance(self):
        cost=self.cost()
        #print(cost)
        return cost[linear_sum_assignment(cost)].sum()
class Cluster():
    def __init__(self, embedding, clusters_num):
        self.embedding=np.array(embedding)
        self.clusters_num=np.intc(clusters_num)

    def clustering(self):#using k-means algorithm
        kmeans=KMeans(self.clusters_num).fit(self.embedding)
        label = np.array(kmeans.labels_)
        return label.T# n by 1 array, n is number of node.









