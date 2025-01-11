import torch
from GPModel import E2EGPModel, CutLoss, BalanceLoss
import numpy as np
from tqdm import tqdm
import time
import numpy as np
import scipy.sparse as sp


def toTorchSparseTensor(adj):
    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    adj_tor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    edge_index = adj_tor.coalesce()
    
    adj_tor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    degrees = torch.sparse.sum(adj_tor, dim=0).to_dense()
    dig_sp = torch.sparse.spdiags(degrees, torch.tensor(0), shape).coalesce()
    
    return edge_index, dig_sp


def partitioning(P, n_clust):
    
    # load graph
    adj = sp.load_npz('./graph/' + P["graph_name"])
    edge_index, dig_sp = toTorchSparseTensor(adj)
    
    # treat all node size as 1.
    node_feature_np = np.ones((adj.shape[0], 1))
    node_feature = torch.from_numpy(node_feature_np).float()
    
    device = torch.device(P["device"])
    node_feature = node_feature.to(device) 
    dig_sp = dig_sp.to(device)
    edge_index = edge_index.to(device)
    
    cut_loss = CutLoss()
    balance_loss = BalanceLoss()
    node_in_channels = node_feature.shape[-1]
    edge_in_channels = edge_index.shape[-1]

    model = E2EGPModel(node_in_channels, edge_in_channels, \
            hidden_channels=P['n_channels'], n_clust=n_clust).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=P['learning_rate'])

    model.train()

    best_loss =  100
    b_lc = 100
    b_lo = 100
    b_assign = None
    lossidx = 0
    blossidx = 0
    
    start = time.time()

    for epoch in tqdm(range(P["es_patience"])):
        optimizer.zero_grad()
        assignment = model(node_feature, edge_index, edge_index, P)
        l_cut = cut_loss(assignment, edge_index, dig_sp)
        l_balance = balance_loss(assignment, node_feature, P)
        loss = P['alpha'] * l_cut + l_balance
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # recording the training process
            lc = l_cut.cpu().detach().numpy() 
            lo = l_balance.cpu().detach().numpy()
            S = assignment.cpu().numpy()
            
            clusters = np.count_nonzero(S, axis=0)
            num_empty_cluster = S.shape[1] - np.count_nonzero(clusters)

            if lc + lo < best_loss:
                best_loss = lc+lo
                b_lc = lc
                b_lo = lo
                blossidx = epoch
                nec= num_empty_cluster
                b_assign = S
                endTime = time.time() - start
            elif epoch - lossidx >= P["patience"] or epoch == P["es_patience"]-1:
                endTime = time.time() - start
                print( "Optimizing " + str(n_clust) + \
                    " partitions " + " stops at ", epoch)
                print("epoch:", blossidx, "best tl:", best_loss, "lc:", b_lc, "lo:", b_lo, "\nTime cost:", endTime, "num_empty_cluster:", nec, flush=True)
                break
    
    print('Edge cut: ', compute_communication_cost(adj, b_assign), '; Partition size list: ', clusters)
    save_path = './output/assignment_'+P['graph_name']
    print('The assignment is saved to: ', save_path)
    np.save(save_path, b_assign)
    return None


def compute_communication_cost(edge_index, part_matrix):
    A_in = edge_index
    resS = part_matrix
    resS = sp.csr_matrix(resS)
    resS_t = resS.transpose()
    A_pooled = (resS_t.dot(A_in)).dot(resS)
    edge_cut = (np.sum(A_in) - np.sum(A_pooled.diagonal()))
    return edge_cut