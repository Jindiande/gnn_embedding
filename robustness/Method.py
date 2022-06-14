# Some methods will be used in Clustering_distance
import numpy as np
import itertools



def center_index_node(label, clusters_num):
    # label = np.array(self.clustering().labels_);
    center_dic = []
    for i in range(clusters_num):
        center_dic.append([])
    for i in range(np.size(label)):
        center_dic[label[i]].append(i)
    center_dic = np.array(list(itertools.zip_longest(*center_dic, fillvalue=-1))).T # padding using -1
    return center_dic  # k by m matrix, m is maximal number in one cluster



def non_overlap_num_one_diemnsion(inte):# intersection operate on 1-d array, return number of set x that not in x\cap y, here inte=[x,y]
    #print(np.shape(inte))
    [x, y] = np.array_split(inte,2)
    num_x=len(x[x!=-1])
    intersection = np.intersect1d(x, y)
    num_inte=len(intersection[intersection!=-1])
    #print(intersection)
    #padded_intersection[:intersection.shape[0]] = intersection
    return num_x-num_inte


def non_overlap_num_one_axis(a, b, axis):# intersection over a and b along one axis using non_overlap_num_one_diemnsion()
    return np.apply_along_axis(non_overlap_num_one_diemnsion, axis, np.concatenate((a, b),axis=1));