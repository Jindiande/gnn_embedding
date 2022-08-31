#GCN from torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCN, GAT

class GCN_naive(nn.Module):# naive GCN model
    def __init__(self,in_channels,hidden_channels,num_layers,out_channels):
        super(GCN_naive, self).__init__()
        self.gcn=GCN(in_channels,hidden_channels,num_layers,out_channels)
        #self.lin=nn.Linear(hidden_channels,out_channels)
        #self.softmax = nn.Softmax(1)
    def forward(self,x,edge_index):
        return self.gcn(x,edge_index)
class GAN_naive(nn.Module):# naive GATN model
    def __init__(self,in_channels,hidden_channels,num_layers,out_channels):
        super(GAN_naive, self).__init__()
        self.gan=GAT(in_channels,hidden_channels,num_layers,out_channels)
        #self.softmax = nn.Softmax(1)
    def forward(self, x, edge_index):
        return self.gan(x, edge_index)

# edge_index = torch.tensor([[0, 2, 1, 0, 3],
#                            [3, 1, 0, 1, 2]], dtype=torch.long)
# x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
# in_channels,out_channels,num_layers=x.size(dim=1),x.size(dim=1),2
# model=GCN_naive(in_channels,out_channels,num_layers)
#
# model.eval()
# print(model(x,edge_index))
