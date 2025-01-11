import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv
from torch.nn import Sequential, Linear, ReLU
operator_datasets = ['bert_l-3_inference', 'bert_l-6_inference', 'bert_l-12_inference', 'resnet50_inference']



class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, aggregator=None, num_layers=None):
        super(GraphGCN, self).__init__()
        # self.convs = torch.nn.ModuleList()
        # self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator, normalize=True))
        self.conv = GCNConv(in_channels, hidden_channels, normalize=True)
        # for i in range(num_layers - 1):
        #     self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator, normalize=True))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


class CutLoss(nn.Module):
    def __init__(self):
        super(CutLoss, self).__init__()

    def forward(self, assignment, adj_sp, d, P):
        # define the loss
        n_clust = assignment.shape[1]
        StAS = torch.matmul(torch.matmul(torch.transpose(assignment, 0,1), adj_sp), assignment)
        # degrees = wighted_adj.sum(dim=0)
        # d = torch.diag(degrees)
        StDS = torch.matmul(torch.matmul(torch.transpose(assignment, 0,1), d), \
            assignment) + 0.001 * torch.eye(n_clust).to(assignment.get_device())
        # compute the determinant
        det = torch.det(StDS)
        # check if the determinant is zero
        if det == 0:
            print("Matrix StDS is singular !")
            print(StDS)
        ncut = torch.matmul(StAS, torch.linalg.inv(StDS))
        cut_loss = -(torch.trace(ncut) / n_clust)
        return cut_loss


class BalanceLoss(nn.Module):
    def __init__(self):
        super(BalanceLoss, self).__init__()

    def forward(self, assignment, X, P):
        # define the loss
        S = assignment
        n_clust = assignment.shape[1]
        # orthogonality loss
        I_S = torch.eye(n_clust).to(assignment.get_device())
        mean_X = torch.sum(X) / n_clust
        
        if P['bigG']:
            X = X.to(torch.device('cpu'))
            dig_X = torch.sparse.spdiags(X.squeeze(1), torch.tensor(0), (assignment.shape[0], assignment.shape[0]))
            dig_X = dig_X.to(assignment.get_device())
        else:
            dig_X = torch.diag(X.squeeze(1))
        StX = torch.matmul(S.t(), dig_X)
        StXS = torch.matmul(StX, S) + 0.001 * I_S
        multipilers = 1. / ((n_clust - 1) * mean_X)
        if P['dataset'] in operator_datasets:
            lo_mat = multipilers * (StXS - mean_X * I_S)
        else:
            means_list = []
            sumX = torch.sum(X)
            while sumX%n_clust != 0:
                num_nodes = torch.ceil(sumX/n_clust)
                means_list.append(num_nodes)
                sumX -= num_nodes
                n_clust -= 1
            for idx in range(n_clust):
                means_list.append(sumX/n_clust)
            mean_diag = torch.diag(torch.FloatTensor(means_list)).to(torch.device(P["device"]))
            lo_mat = multipilers * (StXS - mean_diag)
        ortho_loss = torch.linalg.matrix_norm(lo_mat)
        return ortho_loss


class EnhancedSoftmax(nn.Module):
    def __init__(self):
        super(EnhancedSoftmax, self).__init__()
        self.softmax0 = torch.nn.Softmax(dim=1)
        self.softmax1 = torch.nn.Softmax(dim=1)

    def forward(self, input):
        output = self.softmax0(input)
        eps = 1e-20
        temperature = 0.1
        enhance_U = -torch.log(-torch.log(output + eps) + eps)
        y = output + enhance_U
        output = self.softmax1(y / temperature)
        return output


class TransferOnehot(nn.Module):
    def __init__(self):
        super(TransferOnehot, self).__init__()

    def forward(self, Xsoft, P):
        # implementation 1
        # # Find the maximum value in each row
        # max_values, _ = torch.max(input, dim=1, keepdim=True)
        # # Create a binary mask where the maximum value in each row is set to 1 and all other values are set to 0
        # mask = torch.where(input == max_values, torch.tensor(1).to(P["device"]), torch.tensor(0).to(P["device"]))

        # implementation 2
        # Create a tensor of zeros with the same shape as x
        mask = torch.zeros_like(Xsoft)
        # Find the indices of the maximum value in each row
        max_indices = torch.argmax(Xsoft, dim=1).to(Xsoft.get_device())
        max_indices = max_indices.unsqueeze(1)
        # Set the first maximum value in each row to 1
        mask = mask.scatter_(1, max_indices, 1)

        with torch.no_grad():
            temp = mask - Xsoft
        output_S = temp + Xsoft
        
        # print(input)
        # print(torch.count_nonzero(output_S, dim=0))
        return output_S
