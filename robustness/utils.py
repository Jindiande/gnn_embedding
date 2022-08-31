# utils may be used in data preprocessing/training/cluster evaluation
# reuse some code from pygcn/pygcn/utils
import numpy as np
import scipy.sparse as sp
import torch
import os.path as osp
import os


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def normalize_col(mx):
    """column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(0))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = sp.csr_matrix(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels_onehot = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    #print("edges size",np.shape(edges))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels_onehot.shape[0], labels_onehot.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print(edges.shape)
    # print(adj.todense()[edges[0,1],edges[0,0]])
    adj=  adj + sp.eye(adj.shape[0])
    features_norm = normalize(features)
    adj_norm = normalize(adj)

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    features=torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels_onehot)[1])
    adj_norm_torch = sparse_mx_to_torch_sparse_tensor(adj_norm)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return sparse_mx_to_torch_sparse_tensor(adj), adj_norm_torch,torch.LongTensor(edges).T, features, labels,labels_onehot

def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()
        adj=loader['adj_matrix']
        node_attr=torch.FloatTensor(np.array(loader['node_attr'].todense()))
        node_label=torch.LongTensor(loader['node_label'])
        node_label_onehot = encode_onehot(loader['node_label'])
        colnorm_node_label_onehot=normalize_col(node_label_onehot)
        row_index = adj.indptr
        column_index = adj.indices
        node_num=adj.shape[0]
        edge_num=len(adj.nonzero()[0])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj=  adj + sp.eye(adj.shape[0])
        edge_list = np.zeros((2, edge_num))
        edge_index = 0
        for node_index in range(node_num):
            for j in range(len(column_index[row_index[node_index]:row_index[node_index + 1]])):
                edge_list[0, edge_index] = node_index
                edge_list[1, edge_index] = np.array(column_index[row_index[node_index]:row_index[node_index + 1]])[j]
                edge_index += 1
        return sparse_mx_to_torch_sparse_tensor(adj),torch.LongTensor(edge_list), node_attr, node_label, node_label_onehot,colnorm_node_label_onehot
    else:
        raise ValueError(f"{filepath} doesn't exist.")

