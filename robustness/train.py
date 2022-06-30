#pytorch version implementation of paper Tsitsulin, Anton, et al. "Graph clustering with graph neural networks." arXiv preprint arXiv:2006.16904 (2020).

import time
import argparse

import numpy
import numpy as np
#import scipy.sparse as sp
import torch
#import torch.nn.functional as F
import torch.optim as optim
from model import  GCN_naive
def loss_cal(adj,assignment,features,args):
    degree = torch.sparse.sum(adj, 0).to_dense().reshape(-1, 1)  # d n*1
    edge_num = 2 * torch.sum(degree, 0)
    M1 = torch.mm(assignment.T, torch.sparse.mm(adj, assignment))  # C^T*A*C
    degree_norm = torch.div(degree, edge_num)  # normlized degree
    M2 = assignment.T.mm(degree_norm).mm(degree_norm.T).mm(assignment)
    loss1 = -torch.div(torch.trace(M1 - M2), edge_num)
    loss2 = torch.sqrt(torch.tensor(args.cluster_num)).mul(
        torch.div(torch.linalg.vector_norm(torch.sum(assignment, 0)), features.size(0))) - 1  # sq{k}/n
    loss = torch.add(loss1, loss2)
    return loss,loss1,loss2


def train_and_eval_model(args,adj,edges,features):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data

    in_channels, out_channels, num_layers = features.size(1), args.cluster_num, args.num_layers
    # model=GCN_naive(in_channels,out_channels,num_layers) if args.model=="GCN"
    model = GCN_naive(in_channels, out_channels, num_layers)  # input/hidden/output

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
        assignment = model(features, edges)  # C n*k
        # degree = torch.sparse.sum(adj, 0).to_dense().reshape(-1, 1)  # d n*1
        # edge_num = 2 * torch.sum(degree, 0)
        # M1 = torch.mm(assignment.T, torch.sparse.mm(adj, assignment))  # C^T*A*C
        # degree_norm = torch.div(degree, edge_num)  # normlized degree
        # M2 = assignment.T.mm(degree_norm).mm(degree_norm.T).mm(assignment)
        # loss1 = -torch.div(torch.trace(M1 - M2), edge_num)
        # loss2 = torch.sqrt(torch.tensor(args.cluster_num)).mul(
        #     torch.div(torch.linalg.vector_norm(torch.sum(assignment, 0)), features.size(0))) - 1  # sq{k}/n
        # loss = torch.add(loss1, loss2)
        loss,_,_=loss_cal(adj,assignment,features,args)
        loss.backward()
        optimizer.step()
        # if (epoch % 100 == 0):
        #     # print("loss1 grad",model.gcn.weight.grad)
        #     print(epoch)
        #     print("loss1=", loss1.item(), "loss2=", loss2.item())
    model.eval()
    assignment = model(features, edges)
    #print(assignment)
    _,loss1,loss2=loss_cal(adj,assignment,features,args)
    label=np.argmax(assignment.cpu().detach().numpy(),axis=1)# n*1
    return label,loss1,loss2




