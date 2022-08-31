import time
import argparse
import numpy as np
#import scipy.sparse as sp
import torch
from train import train_and_eval_model
from Clustering_distance import Cluster_distance
from utils import load_data, load_npz,normalize_col,encode_onehot
from sklearn.metrics.cluster import normalized_mutual_info_score
import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cluster_num', type=int, default=7,
                    help='number of clusters.')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in GNN.')
parser.add_argument('--out_channels', type=int, default=16,
                    help='output feature size of model')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model_type', type=str, default="GCN",
                    help='model catagory, GCN, GAN or null')
parser.add_argument('--if_random_feature', type=bool, default=False,
                    help='if using random node attribute.')
parser.add_argument('--loss_type', type=str, default="modularity",
                    help='type of loss in training, now support modularity, ratio_mincut, n_mincut, naive_dec and modu_dec')
parser.add_argument('--dataset_type', type=str,default="cora",
                    help='data set, support cora, citeseer, pubmed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
   print("Using cuda")
def stability_clusternum(start,end):# return stability for different cluster number, range from [start, end]
    print("stability test ")
    cluster_number_list=np.arange(start,end,10)
    run_time=5
    stability_list=np.zeros(cluster_number_list.shape)
    datadir = "/tmp/cluster/robustness/GraphData/datasets/"
    adj, edges, features, _, _, _ = load_npz(datadir + args.dataset_type + ".npz")
    nodes_num=features.size(0)# varies for different dataset
    #args.if_random_feature=False
    for index1 in range(len(cluster_number_list)):
        distance_sum=0
        label_list=np.zeros([run_time,nodes_num])
        args.cluster_num=cluster_number_list[index1]
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        if (args.if_random_feature):
            features = torch.randn(features.size())  # dense random node attribute
        for index2 in range(run_time):
            # if (args.if_random_feature):
            #     features = torch.randn(features.size())  # dense random node attribute
            #     #print(features[0:10,0])
            label,_= train_and_eval_model(args,adj,edges,features)
            #print(label)
            label_list[index2,:]=label
        #print("No.", index1)
        for i in range(run_time):
            for j in range(i+1,run_time):
                cluster_nmi=normalized_mutual_info_score(np.array(label_list[i,:]).astype(int), np.array(label_list[j,:]).astype(int))
                #cluster_distance=Cluster_distance(np.array(label_list[i,:]).astype(int),np.array(label_list[j,:]).astype(int),args.cluster_num)
                distance_sum+=cluster_nmi

        stability_list[index1] = 2*distance_sum/(run_time*(run_time-1))
    datetimeInstance = datetime.datetime.today()
    file = open("./result/"+args.dataset_type+"stability_clusternum_"+"dim="+str(args.out_channels)+"_" + args.loss_type+"_"+ args.model_type+"_"+ str(datetimeInstance) + ".txt", "w")
    for item_index in range(len(stability_list)):
        file.write(str(cluster_number_list[item_index])+" "+str(100*stability_list[item_index])+"%\n")
    return 0
def stability_dimension():# return stability for different cluster number, range from [start, end]
    print("stability test ")
    dime_list=np.arange(2,1000,50)
    # cluster_number_list=np.arange(start,end,10)
    run_time=5
    stability_list=np.zeros(dime_list.shape)
    datadir = "/tmp/cluster/robustness/GraphData/datasets/"
    adj, edges, features, _, _, _ = load_npz(datadir + args.dataset_type + ".npz")
    nodes_num=features.size(0)# varies for different dataset
    #args.if_random_feature=False
    for index1 in range(len(dime_list)):
        args.out_channels = dime_list[index1]
        distance_sum=0
        label_list=np.zeros([run_time,nodes_num])
        args.cluster_num=7
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        if (args.if_random_feature):
            features = torch.randn(features.size())  # dense random node attribute
        for index2 in range(run_time):
            # if (args.if_random_feature):
            #     features = torch.randn(features.size())  # dense random node attribute
            #     #print(features[0:10,0])
            label,_= train_and_eval_model(args,adj,edges,features)
            # print(label)
            label_list[index2,:]=label
        #print("No.", index1)
        for i in range(run_time):
            for j in range(i+1,run_time):
                cluster_nmi=normalized_mutual_info_score(np.array(label_list[i,:]).astype(int), np.array(label_list[j,:]).astype(int))
                #cluster_distance=Cluster_distance(np.array(label_list[i,:]).astype(int),np.array(label_list[j,:]).astype(int),args.cluster_num)
                distance_sum+=cluster_nmi

        stability_list[index1] = 2*distance_sum/(run_time*(run_time-1))
    datetimeInstance = datetime.datetime.today()
    file = open("./result/"+args.dataset_type+"stability_dimension_" + args.loss_type+"_"+ args.model_type+"_"+ str(datetimeInstance) + ".txt", "w")
    for item_index in range(len(stability_list)):
        file.write(str(dime_list[item_index])+" "+str(100*stability_list[item_index])+"%\n")
    return 0
# def dimen_test():#test how embedding dimension effect clustering
#     print("dimension test")
#     run_time = 10
#     # adj, _, edges, features, label_onehot = load_data()
#     datadir = "./GraphData/datasets/"
#     adj,edges,features,label=load_npz(datadir+args.dataset_type+".npz")
#     #print(np.array(label_onehot))
#     #nodes_num = features.size(0)  # varies for different dataset
#     #nmi_sum = 0
#     dime_list=np.arange(2,1000,50)
#     nmi_list=np.zeros(len(dime_list))
#     if (args.if_random_feature):
#         features = torch.randn(features.size())  # dense random node attribute
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#     for dim_index in range(len(dime_list)):
#         args.out_channels = dime_list[dim_index]
#         nmi_sum=0
#         for i in range(run_time):
#             label_pred = train_and_eval_model(args, adj, edges, features)
#             #print(label_pred)
#             cluster_nmi = normalized_mutual_info_score(label_pred, label)
#             # #cluster_nmi = Cluster_distance(np.array(label_pred).astype(int),
#             #                                     np.array(label_onehot).astype(int), args.cluster_num)
#             nmi_sum += cluster_nmi
#         nmi_list[dim_index]=nmi_sum/run_time
#         print("dim=", dime_list[dim_index], "Nmi=", nmi_sum / run_time)
#     datetimeInstance = datetime.datetime.today()
#     file = open("./result/Dimentest_NMI"+"_" + args.loss_type+"_"+args.dataset_type+"_" + str(datetimeInstance) + ".txt", "w")
#     for item_index in range(len(nmi_list)):
#         file.write(str(dime_list[item_index])+str(100 * nmi_list[item_index]) + "%\n")
#     return nmi_list

#
def dimen_test_NMI_PIP():#test \|B_g B_g^T-B_p B_p^T\| varing dimension
    print("dimension test loss")
    run_time = 10
    # adj, _, edges, features, label_onehot = load_data()
    datadir = "/tmp/cluster/robustness/GraphData/datasets/"
    adj,edges,features,label,_,norm_label_onehot=load_npz(datadir+args.dataset_type+".npz")
    norm_label_onehot = torch.tensor(norm_label_onehot.todense()).cuda()
    #print(np.array(label_onehot))
    #nodes_num = features.size(0)  # varies for different dataset
    #nmi_sum = 0
    dime_list=np.arange(2,1000,50)
    loss_list=np.zeros(len(dime_list))
    nmi_list=np.zeros(len(dime_list))
    if (args.if_random_feature):
        features = torch.randn(features.size())  # dense random node attribute
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    for dim_index in range(len(dime_list)):
        args.out_channels = dime_list[dim_index]
        loss1=0
        nmi_sum=0
        for i in range(run_time):
            label_pred,_ = train_and_eval_model(args, adj, edges, features)
            label_onehot_pred=encode_onehot(label_pred)
            norm_label_onehot_pred=normalize_col(label_onehot_pred)
            norm_label_onehot_pred=torch.tensor(norm_label_onehot_pred.todense()).cuda()
            loss1+=torch.linalg.matrix_norm(torch.mm(norm_label_onehot,norm_label_onehot.T)-torch.mm(norm_label_onehot_pred,norm_label_onehot_pred.T))
            cluster_nmi = normalized_mutual_info_score(label_pred, label)
            # #cluster_nmi = Cluster_distance(np.array(label_pred).astype(int),
            #                                     np.array(label_onehot).astype(int), args.cluster_num)
            nmi_sum += cluster_nmi
        loss_list[dim_index]=(loss1/run_time)
        nmi_list[dim_index]=(nmi_sum/run_time)
        #print("dim=", dime_list[dim_index], "loss=", loss_list[dim_index], "nmi=",nmi_list[dim_index])
    datetimeInstance = datetime.datetime.today()
    file = open("./result/"+args.dataset_type+"dimen_test_gt_res_innerprod_loss_" + args.loss_type+"_"+ args.model_type+"_"+ str(datetimeInstance) + ".txt", "w")
    for item_index in range(len(loss_list)):
        file.write(str(dime_list[item_index])+" "+str(loss_list[item_index]) + "\n")
    file = open("./result/"+args.dataset_type+"dimen_test_gt_NMI_" + args.loss_type+"_"+ args.model_type+"_"+ str(datetimeInstance) + ".txt", "w")
    for item_index in range(len(nmi_list)):
        file.write(str(dime_list[item_index])+" "+str(nmi_list[item_index]) + "\n")
    return 0

# args.dataset_type="citeseer"
# args.loss_type="naive_enc"
# print(dimen_test())
# args.loss_type="modu_enc"
# print(dimen_test())


args.loss_type="naive_dec"
# stability_clusternum(2, 100)
# stability_dimension()
model_type=["GCN","GAN","null"]
dataset_type=["cora","citeseer"]
for i in dataset_type:
    for j in model_type:
        args.model_type = j
        args.dataset_type = i
        args.out_channels=50
        print(i,j,args.out_channels)
        stability_clusternum(2,100)
        args.out_channels=100
        print(i,j,args.out_channels)
        stability_clusternum(2,100)
        # dimen_test_NMI_PIP()
        # stability_dimension()






#def dimen_test_gt_null_space_lost():#how much null space of gt
#     print("dimension test loss")
#     run_time = 10
#     # adj, _, edges, features, label_onehot = load_data()
#     datadir = "./GraphData/datasets/"
#     adj,edges,features,_,_,norm_label_onehot=load_npz(datadir+args.dataset_type+".npz")
#     norm_label_onehot = torch.tensor(norm_label_onehot.todense()).cuda()
#     #print(np.array(label_onehot))
#     #nodes_num = features.size(0)  # varies for different dataset
#     #nmi_sum = 0
#     dime_list=np.arange(2,1000,50)
#     loss_list=np.zeros(len(dime_list))
#     if (args.if_random_feature):
#         features = torch.randn(features.size())  # dense random node attribute
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#     for dim_index in range(len(dime_list)):
#         args.out_channels = dime_list[dim_index]
#         loss1=0
#         for i in range(run_time):
#             _,feature_learn = train_and_eval_model(args, adj, edges, features)
#             proj_m=torch.eye(feature_learn.size(0)).cuda()-torch.mm(norm_label_onehot,norm_label_onehot.T)
#             loss1+=torch.linalg.matrix_norm(torch.mm(proj_m.double(),feature_learn.double()))/torch.linalg.matrix_norm(feature_learn)
#             #cluster_nmi = normalized_mutual_info_score(label_pred, label)
#             # #cluster_nmi = Cluster_distance(np.array(label_pred).astype(int),
#             #                                     np.array(label_onehot).astype(int), args.cluster_num)
#             # nmi_sum += cluster_nmi
#         loss_list[dim_index]=(loss1/run_time)
#         print("dim=", dime_list[dim_index], "loss=", loss_list[dim_index])
#     datetimeInstance = datetime.datetime.today()
#     file = open("./result/dimen_test_gt_null_space_lost_" + args.loss_type+"_"+args.dataset_type+"_" + str(datetimeInstance) + ".txt", "w")
#     for item_index in range(len(loss_list)):
#         file.write(str(dime_list[item_index])+str(loss_list[item_index]) + "\n")
#     return loss_list
# def prediction():
#     print("prediction")
#     run_time=10
#     adj, _, edges, features,label_onehot = load_data()
#     #print(np.array(label_onehot))
#     nodes_num=features.size(0)# varies for different dataset
#     nmi_sum=0
#     if (args.if_random_feature):
#         features = torch.randn(features.size())  # dense random node attribute
#     for i in range(run_time):
#         label_pred = train_and_eval_model(args, adj, edges, features)
#         print(label_pred)
#         cluster_nmi=normalized_mutual_info_score(label_pred, label_onehot)
#         # #cluster_nmi = Cluster_distance(np.array(label_pred).astype(int),
#         #                                     np.array(label_onehot).astype(int), args.cluster_num)
#         nmi_sum += cluster_nmi
#     # datetimeInstance = datetime.datetime.today()
#     # file = open("NMI" + args.loss_type + str(datetimeInstance) + ".txt", "w")
#     # file.write(str(nmi_sum/run_time))
#     return nmi_sum/run_time
