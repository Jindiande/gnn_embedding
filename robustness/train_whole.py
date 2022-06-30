import time
import argparse
import numpy as np
#import scipy.sparse as sp
import torch
from train import train_and_eval_model
from Clustering_distance import Cluster_distance
from utils import load_data,sparse_mx_to_torch_sparse_tensor
import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cluster_num', type=int, default=7,
                    help='number of clusters.')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in GNN.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default="GCN",
                    help='GNN catagory, default GCN.')
parser.add_argument('--if_random_feature', type=bool, default=True,
                    help='if using random node attribute.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
   print("Using cuda")
def stability(start,end):# return stability for different cluster number, range from [start, end]
    cluster_number_list=np.arange(start,end)
    run_time=10
    nodes_num=2708# varies for different dataset
    stability_list=np.zeros(cluster_number_list.shape)
    adj, _, edges, features, _, = load_data()
    #args.if_random_feature=False
    if(args.if_random_feature):
          features = torch.randn(features.size()) # dense random node attribute
    for index1 in range(len(cluster_number_list)):
        distance_sum=0
        label_list=np.zeros([run_time,nodes_num])
        args.cluster_num=cluster_number_list[index1]
        for index2 in range(run_time):
            label,loss1,loss2 = train_and_eval_model(args,adj,edges,features)
            #print(label)
            label_list[index2,:]=label
        print("No.", index1)
        for i in range(run_time):
            for j in range(i+1,run_time):
                cluster_distance=Cluster_distance(np.array(label_list[i,:]).astype(int),np.array(label_list[j,:]).astype(int),args.cluster_num)
                distance_sum+=cluster_distance.distance()
        stability_list[index1] = 2*distance_sum/(run_time*(run_time-1))
    datetimeInstance = datetime.datetime.today()
    file = open(str(datetimeInstance) + ".txt", "w")
    for item_index in range(len(stability_list)):
        file.write(str(cluster_number_list[item_index])+str(100*stability_list[item_index]/nodes_num)+"%\n")
    return stability_list
print(stability(10,100))