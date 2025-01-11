import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, aggregator, num_layers):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=0.05)
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator, normalize=True))
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator, normalize=True))

    def forward(self, x, edge_index):
        out = x
        for conv in self.convs[:-1]:
            out = self.relu(self.dropout(conv(out, edge_index)))
        out = self.convs[-1](out, edge_index)
        return out
    

class E2EGPModel(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, pos_in_channels=20, hidden_channels=16, n_clust=16):
        super(E2EGPModel, self).__init__()
        self.node_linear = torch.nn.Linear(in_features=node_in_channels, \
            out_features=hidden_channels)
        self.edge_linear = torch.nn.Linear(in_features=edge_in_channels, \
            out_features=hidden_channels)
        self.pos_linear = torch.nn.Linear(in_features=pos_in_channels, \
            out_features=hidden_channels)
        
        self.num_sage_layer = 1
        
        self.shared_sage = GraphSAGE(in_channels=2*hidden_channels, \
            hidden_channels=2*hidden_channels, aggregator='add', num_layers=self.num_sage_layer)
        
        self.sum_sage = GraphSAGE(in_channels=2*hidden_channels, \
            hidden_channels=n_clust, aggregator='add', num_layers=self.num_sage_layer)
        
        self.relu = torch.nn.ELU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.getOnehot = TransferOnehot()

    def forward(self, nodefeature, edge_feature, edge_index, position_feature=None):
        node_dim = self.node_linear(nodefeature)
        edge_dim = self.edge_linear(edge_feature)
        
        # if you have position feature, you may need change dimension and code accordingly:
        # pos_dim = self.pos_linear(position_feature)
        # whole_embed = torch.cat((node_dim, edge_dim, pos_dim), 1)
        whole_embed = torch.cat((node_dim, edge_dim), 1)
        
        reduced_embed = self.shared_sage(x=whole_embed, edge_index=edge_index)
        reduced_embed = self.relu(reduced_embed)

        clust_embed = self.sum_sage(x=reduced_embed, edge_index=edge_index)
        
        soft_S = self.logsoftmax(clust_embed)
        
        hard_S = self.getOnehot(soft_S)

        return hard_S


class CutLoss(nn.Module):
    def __init__(self):
        super(CutLoss, self).__init__()

    def forward(self, assignment, adj_sp, d):
        # define the loss
        n_clust = assignment.shape[1]
        StAS = torch.matmul(torch.matmul(torch.transpose(assignment, 0,1), adj_sp), assignment)
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
        

        X = X.to(torch.device('cpu'))
        dig_X = torch.sparse.spdiags(X.squeeze(1), torch.tensor(0), (assignment.shape[0], assignment.shape[0]))
        dig_X = dig_X.to(assignment.get_device())

        # genereate perfect balance partition size
        means_list = []
        sumX = torch.sum(X)
        while sumX%n_clust != 0:
            num_nodes = torch.ceil(sumX/n_clust)
            means_list.append(num_nodes)
            sumX -= num_nodes
            n_clust -= 1
        for idx in range(n_clust):
            means_list.append(sumX/n_clust)
        
        # calculate balance loss
        StX = torch.matmul(S.t(), dig_X)
        StXS = torch.matmul(StX, S) + 0.001 * I_S
        multipilers = 1. / ((n_clust - 1) * mean_X)
        
        mean_diag = torch.diag(torch.FloatTensor(means_list)).to(torch.device(P["device"]))
        lo_mat = multipilers * (StXS - mean_diag)
        ortho_loss = torch.linalg.matrix_norm(lo_mat)
        return ortho_loss


class TransferOnehot(nn.Module):
    def __init__(self):
        super(TransferOnehot, self).__init__()

    def forward(self, Xsoft):

        mask = torch.zeros_like(Xsoft)
        # Find the indices of the maximum value in each row
        max_indices = torch.argmax(Xsoft, dim=1).to(Xsoft.get_device())
        max_indices = max_indices.unsqueeze(1)
        # Set the first maximum value in each row to 1
        mask = mask.scatter_(1, max_indices, 1)

        with torch.no_grad():
            temp = mask - Xsoft
        output_S = temp + Xsoft

        return output_S


