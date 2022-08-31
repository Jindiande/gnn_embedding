#pytorch version implementation of paper Tsitsulin, Anton, et al. "Graph clustering with graph neural networks." arXiv preprint arXiv:2006.16904 (2020).

import time
import argparse
import sys
import numpy
import numpy as np
#import scipy.sparse as sp
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import  GCN_naive, GAN_naive
from Clustering_loss import LossClass
from Clustering_distance import Cluster




def train_and_eval_model(args,adj,edges,features):
    # print(args.loss_type)
    # Load data
    if (args.loss_type =="naive_dec" or args.loss_type =="modu_dec"):
        # print("encoder")
        in_channels, hidden_channels, num_layers,out_channels = features.size(1), 32, args.num_layers,args.out_channels
        # model=GCN_naive(in_channels,out_channels,num_layers) if args.model=="GCN"
        if args.cuda:
            #print("Using cuda")
            features = features.cuda()
            edges = edges.cuda()
            adj = adj.cuda()
        if (args.model_type == 'null'):
            features=features[:,0:args.out_channels]
            features=Variable(features,requires_grad=True)
            optimizer = optim.Adam([features],
                                   lr=args.lr, weight_decay=args.weight_decay)
        else:
            if(args.model_type == 'GCN'):
                 model = GCN_naive(in_channels, hidden_channels, num_layers,out_channels)  # input/hidden/output
            else:
                 model = GAN_naive(in_channels, hidden_channels, num_layers, out_channels)  # input/hidden/output
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
            model.train()  # training session begin
        if args.model_type!= 'null': model.cuda()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            if(args.model_type=='null'):
                features_new=features
            else:
                features_new = model(features, edges)  # C n*k
            Loss = LossClass(adj, args, features=features_new, assignment=None)
            if (args.loss_type == "modu_dec"):
                loss_fun=Loss.modu_encoder_decoder
            else:
                loss_fun = Loss.naive_encoder_decoder
            loss=loss_fun()
            loss.backward()
            optimizer.step()
            # if (epoch % 100 == 0):
            #     print(loss.item())
        cluster=Cluster(features_new.cpu().detach().numpy(), args.cluster_num)
        label=cluster.clustering()
        return label, features_new



    elif ( args.loss_type =="modularity" or args.loss_type =="ratio_mincut" or args.loss_type =="n_mincut"):
        in_channels, hidden_channels, num_layers,out_channels = features.size(1), features.size(1), args.num_layers,args.cluster_num
        # model=GCN_naive(in_channels,out_channels,num_layers) if args.model=="GCN"
        # print(in_channels, hidden_channels, num_layers,out_channels,)
        model = GCN_naive(in_channels, hidden_channels, num_layers,out_channels)  # input/hidden/output

        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        # features=torch.zeros(features.size()).cuda()

        # features=sparse_mx_to_torch_sparse_tensor(sp.random(features.size(0),features.size(1),density=0.1)).to_dense()
        # features=torch.nn.functional.normalize(features, p=2.0, dim = 1)
        # print(features[0:1,1:100])
        if args.cuda:
            #print("Using cuda")
            model.cuda()
            features = features.cuda()
            edges = edges.cuda()
            adj = adj.cuda()

        model.train()  # training session begin
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            assignment = F.softmax(model(features, edges),dim=1)  # C n*k
            Loss = LossClass(adj, args, features=None, assignment=assignment)
            if(args.loss_type == "modularity"):
                loss_fun=Loss.modularity
            elif(args.loss_type == "ratio_mincut"):
                loss_fun=Loss.ratio_mincut
            else:
                loss_fun=Loss.n_mincut
            loss, loss1, loss2 = loss_fun()
            loss.backward()
            optimizer.step()
            # if (epoch % 100 == 0):
            #     # print("loss1 grad",model.gcn.weight.grad)
            #     #print(epoch,"loss=",loss.item())
            #     print("loss1=", loss1.item(), "loss2=", loss2.item())
        model.eval()
        assignment = F.softmax(model(features, edges),dim=1)
        #print(assignment)
        #_, loss1, loss2 = loss_fun(adj, assignment, features, args)
        #print(assignment.size())
        label = np.argmax(assignment.cpu().detach().numpy(), axis=1)  # n*1

    else:
        sys.exit("Loss not support")


    return label#,loss1,loss2




