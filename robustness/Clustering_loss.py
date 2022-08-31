#import argparse
#import numpy as np
#import scipy.sparse as sp
import torch
#import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

class LossClass():
    def __init__(self,adj,args,features=None,assignment=None):
        self.adj=adj
        self.assignment=assignment
        if(features!=None):
            self.features=features
        if(assignment!=None):
            self.assignment=assignment
        self.args=args
        self.fudge = 1e-7
    def modularity(self):#modularity and regularization loss
        degree = torch.sparse.sum(self.adj, 0).to_dense().reshape(-1, 1)  # d n*1
        edge_num = torch.sum(degree, 0)#2m
        M1 = torch.mm(self.assignment.T, torch.sparse.mm(self.adj, self.assignment))  # C^T*A*C
        degree_norm = torch.div(degree, edge_num)  # normlized degree
        M2 = self.assignment.T.mm(degree).mm(degree_norm.T).mm(self.assignment)#C^T d d^T C/2m
        loss1 = -torch.div(torch.trace(M1 - M2), edge_num)
        loss2 = torch.sqrt(torch.tensor(self.args.cluster_num)).mul(
            torch.div(torch.linalg.vector_norm(torch.sum(self.assignment, 0)), self.assignment.size(0))) - 1  # sq{k}/n
        loss = torch.add(loss1, loss2)
        return loss, loss1, loss2
    def ratio_mincut(self):#ratio mincut and regularization loss
        degree = torch.sparse.sum(self.adj, 0).to_dense()  # d n*1
        edge_num = torch.sum(degree)#2*m
        self.assignment=torch.mm(self.assignment,torch.diag(torch.div(torch.ones(self.assignment.size(1)).cuda(),torch.linalg.vector_norm(self.assignment,dim=0,ord=2))))
        M1 = torch.mm(self.assignment.T, torch.sparse.mm(self.adj, self.assignment))  # C^T*A*C
        M2= torch.mm(self.assignment.T, torch.sparse.mm(torch.diag(degree).to_sparse(),self.assignment))# C^T*D*C
        loss1=torch.div(torch.trace(M2 - M1), edge_num)
        loss2=torch.linalg.matrix_norm(torch.mm(self.assignment.T,self.assignment)-torch.eye(self.assignment.size(1)).cuda(),ord='fro')
        # loss2 = torch.sqrt(torch.tensor(self.args.cluster_num)).mul(
        #     torch.div(torch.linalg.vector_norm(torch.sum(self.assignment, 0)), self.assignment.size(0))) - 1  # sq{k}/n
        loss = torch.add(loss1, loss2)
        return loss, loss1, loss2
    def n_mincut(self):#N mincut and regularization loss
        degree = torch.sparse.sum(self.adj, 0).to_dense()  # d n*1
        edge_num = torch.sum(degree)  # 2*m
        temp=torch.mm(torch.diag(degree),self.assignment)# n*k
        self.assignment = torch.mm(self.assignment,torch.diag(torch.div(torch.ones(self.assignment.size(1)).cuda(),torch.linalg.vector_norm(temp,dim=0,ord=2))))
        M1 = torch.mm(self.assignment.T, torch.sparse.mm(self.adj, self.assignment))  # C^T*A*C
        M2 = torch.mm(self.assignment.T, torch.sparse.mm(torch.diag(degree).to_sparse(), self.assignment))  # C^T*D*C
        loss1 = torch.div(torch.trace(M2 - M1), edge_num)
        loss2=torch.linalg.matrix_norm(M2-torch.eye(M2.size(0)).cuda(),ord='fro')
        loss = torch.add(loss1, loss2)
        return loss, loss1, loss2
    def naive_encoder_decoder(self):
        #print(self.features.size())
        adj_out=torch.mm(self.features,self.features.T)+self.fudge
        pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - torch.sparse.sum(self.adj)) / torch.sparse.sum(self.adj)
        # print(self.adj.to_dense().view(-1,1))
        # print(adj_out.view(-1,1))
        # print(self.adj.type())
        norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - torch.sparse.sum(self.adj)) * 2)
        loss=norm*F.binary_cross_entropy_with_logits(adj_out,self.adj.to_dense(),reduction='mean',pos_weight=pos_weight)
        #print(loss.size())
        return loss
    def modu_encoder_decoder(self):
        adj_out = torch.mm(self.features, self.features.T) + self.fudge
        degree = torch.sparse.sum(self.adj, 0).to_dense().reshape(-1, 1)  # d n*1
        edge_num = torch.sum(degree, 0)#2m
        degree_norm = torch.div(degree, edge_num)  # normlized degree
        pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - torch.sparse.sum(self.adj)) / torch.sparse.sum(self.adj)
        # print(self.adj.to_dense().view(-1,1))
        # print(adj_out.view(-1,1))
        norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - torch.sparse.sum(self.adj)) * 2)
        loss = norm*F.binary_cross_entropy_with_logits(adj_out, self.adj.to_dense()-degree.mm(degree_norm.T), reduction='mean',pos_weight=pos_weight)
        return loss



