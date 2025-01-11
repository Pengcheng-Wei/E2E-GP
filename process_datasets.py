import os
import json
import networkx as nx
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.sparse.linalg import eigs
import math, sys
from skimage import graph
from skimage.io import imread
from skimage import segmentation, color

operator_datasets = ['bert_l-3_inference', 'bert_l-6_inference', 'bert_l-12_inference', 'resnet50_inference']
citation_datasets = ['cora', 'citeseer', 'pubmed']
pyg_dataset = ['soc-LiveJournal1', 'com-lj.ungraph.txt', 'com-youtube', 'git_web_ml', 'twitch_gamers', 'com-orkut', 'BlogCatalog', 'AmazonProducts', 'PT', 'ES', 'Flickr', 'yelp']
Walshaw_dataset = ['add20', 'data','finan512', 'fe_rotor', 'wing', "3elt", 'bcsstk33', 'crack', 'vibrobox', 'cs4', 'wing_nodal', 'bcsstk30']
rootDatasetPath = '/mnt/data/pengcheng/datasets/e2egpDatasets/'

# nohup jupyter notebook --allow-root > ~/.jupyter/jupyter.log 2>&1 &
# bert-l3: nodes=235, edges=250
# bert-l12: nodes=783, edges=843
def load_operator_data(filename):
    with open(rootDatasetPath + 'OperatorGraphs/' + filename + '.json', 'r') as load_f:
        load_dict = json.load(load_f)
    nodes_list = load_dict['nodes']
    edges_list = load_dict['edges']
    # generate undirected graph
    graph = nx.Graph()
    g_elements = []
    n_id = []
    for item in edges_list:
        sourceId = int(item['sourceId'])
        destId = int(item['destId'])
        cost = int(item['cost'] * 1e4) + 1
        # cost = math.atan(float(item['cost']) * 1000000)*100
        g_elements.append((sourceId, destId, {'weight': cost}))
        if sourceId not in n_id:
            n_id.append(sourceId)
        if destId not in n_id:
            n_id.append(destId)
    # get raw adj
    temp_g = nx.Graph()
    temp_g.add_edges_from(g_elements)
    adj = nx.adjacency_matrix(temp_g)
    adj = sp.csr_matrix(adj)#.tolil()
    #adj = adj.todense()

    # normalize edges weights
    # edges_weights = np.array([g_elements[i][2]['weight'] for i in range(len(g_elements))])
    # # print(sum(edges_weights))
    # for i in range(len(g_elements)):
    #     g_elements[i][2]['weight'] = g_elements[i][2]['weight'] / np.linalg.norm(edges_weights)
    # graph.add_edges_from(g_elements)
    # generate adj matrix A with weights (edge cost)
    # adj_normalized = nx.adjacency_matrix(graph)
    # adj_normalized = sp.csr_matrix(adj_normalized).tolil().todense()

    X = np.arange(len(nodes_list), dtype=float).reshape(len(nodes_list), 1)
    for item in nodes_list:
        id = int(item['id'])
        pos = n_id.index(id)
        X[pos, 0] = int(item['size'])/10000 + 1

    X = sp.csr_matrix(X).tolil().todense()
    X = np.asarray(X)
    # X_normalized = sp.csr_matrix(X_normalized).tolil().todense()
    return adj, X

def load_unweighted_partition_data(filename):
    adjnpzPath = rootDatasetPath + 'partitionData/graphs/' + filename + '.npz'
    if os.path.isfile(adjnpzPath):
        print('Loading the graph', filename, '...', flush=True)
        adj = sp.load_npz(adjnpzPath)
        node_features = np.ones((adj.shape[0], 1))
        # node_features = sp.csr_matrix(node_features).tolil().todense()
    else:
        print('Constructing the graph', filename, '...', flush=True)
        filepath = rootDatasetPath + 'partitionData/graphs/' + filename + '.graph'
        graphfile = open(filepath)
        num_nodes = 0
        cnt = 0
        for line in graphfile:
            if cnt==0:
                num_nodes, _ = map(int, line.split())
                adj = np.zeros((num_nodes, num_nodes), dtype=float)
            else:
                nerb_list = map(int, line.split())
                for nerb in nerb_list:
                    adj[cnt-1][nerb-1] = 1
                    pass
            cnt += 1
        node_features = np.ones((num_nodes, 1))
        # node_features = sp.csr_matrix(node_features).tolil().todense()
        adj = sp.csr_matrix(adj)
        sp.save_npz(rootDatasetPath + 'partitionData/graphs/'+filename+'.npz', adj)
        # np.savez_compressed(rootDatasetPath + 'partitionData/graphs/'+filename+'.npz', dataset=adj)
    return adj, node_features

def load_pyg_data(filename, P=None):
    npzPath = rootDatasetPath + 'pyg_datasets/'
    if False:    
    # if P['use_coarsen']:    
        adj = np.load(npzPath + filename + '-adj.npz')['dataset']
        print('Reduction shape:', adj.shape, flush=True)
        node_features = np.load(npzPath + filename + '-ns.npz')['dataset']
    else:
        dataPath = npzPath + filename + '.npz'
        adj = sp.load_npz(dataPath)
        node_features = np.ones((adj.shape[0], 1))
        # node_features = sp.csr_matrix(node_features).tolil().todense()
    return adj, node_features


def computeAvgDegree():
    filesdir = '/home/pengcheng/documents/projects/e2egp-torch/data/partitionData/graphs'
    flielist = os.listdir(filesdir)
    for dataset in flielist:
        dataset = dataset.replace('.graph', '')
        adj, node_features = load_unweighted_partition_data(dataset)
        num_nodes = node_features.shape[0]
        if num_nodes < 20000:
            # calculate the degree of each node
            degree = np.sum(adj, axis=1)
            # calculate the average degree of nodes
            avg_degree = np.mean(degree)
            print(dataset, num_nodes, avg_degree)


def prepareInput4Torch(P):
    
    # load data
    filename = P["dataset"]

    if P["dataset"] in P['Walshaw_data']:
        adj, X = load_unweighted_partition_data(P["dataset"])
    elif P["dataset"] in pyg_dataset:
        adj, X = load_pyg_data(filename, P)
        selfloopNum = adj.max()
        diags = [selfloopNum for i in range(adj.shape[0])]
        adj.setdiag(diags)
    else:
        adj, X = load_operator_data(filename=P["dataset"])
        diags = [0 for i in range(adj.shape[0])]
        adj.setdiag(diags)


    nodefeature = np.copy(X)
    nodefeature = torch.from_numpy(nodefeature).float()
    X = torch.from_numpy(X).float()
    
    if nodefeature.shape[0] < 10000:
        P['use_pos_fea'] = True
        P['use_pca'] = True
    
    cos_dist_mat = None
    edge_feature = None
    if P['use_pos_fea']:
        cos_dist_mat = getShortestPathMatrixDijkstra(P['dataset'], adj)
    
    if P['bigG']:
        adj = sp.coo_matrix(adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape
        adj_tor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        edge_index = adj_tor.coalesce()
        
        degrees = torch.sparse.sum(adj_tor, dim=0).to_dense()
        dig_sp = torch.sparse.spdiags(degrees, torch.tensor(0), shape).coalesce()
        


    return nodefeature, edge_feature, edge_index, adj_sp, X, cos_dist_mat, dig_sp

def getShortestPathMatrixDijkstra(dataset, adj):
    
    path = rootDatasetPath + 'pre-calculated/' + dataset + ".npz"
    if os.path.isfile(path):
        print('Loading shotest path')
        cos__dist_mat = np.load(path)['cos__dist_mat']
        cos__dist_mat = torch.from_numpy(cos__dist_mat).float()
    else:
        print('Calculating shotest path')
        dist_matrix = shortest_path(csgraph=adj, directed=False)
        dia = np.max(dist_matrix)
        nor_dist_mat = dist_matrix/dia
        pi = torch.Tensor(([math.pi]))
        cos__dist_mat = torch.cos(pi*nor_dist_mat).float()
        np.savez(path, cos__dist_mat=cos__dist_mat.detach().numpy())
    # print(cos__dist_mat)
    return cos__dist_mat

def get_pca_mat(dataset, adj, k, usage):
    if usage == "shortestPath":
        shortestPath_path = rootDatasetPath + 'pre-calculated/shortestPath_' + dataset + "_pca_" + str(k) + ".npz"
        if os.path.isfile(shortestPath_path):
            print('Loading position feature...')
            X_reconstructed = np.load(shortestPath_path)['X_reconstructed']
            X_reconstructed = torch.from_numpy(X_reconstructed).float()
        else:
            print('Calculating position feature...')
            X = getShortestPathMatrixDijkstra(dataset, adj)
            X_reconstructed = pca_reduce(X, k)
            np.savez(shortestPath_path, X_reconstructed=X_reconstructed.detach().numpy())
    elif usage == "adjacentMat":
        adj_pca_path = rootDatasetPath + 'pre-calculated/adjacentMat_' + dataset + "_pca_" + str(k) + ".npz"
        if os.path.isfile(adj_pca_path):
            print('Loading adj feature...')
            X_reconstructed = np.load(adj_pca_path)['X_reconstructed']
            X_reconstructed = torch.from_numpy(X_reconstructed).float()
        else:
            print('Calculating adj feature...')
            X = adj
            # X = torch.from_numpy(adj).float()
            X_reconstructed = pca_reduce(X, k)
            np.savez(adj_pca_path, X_reconstructed=X_reconstructed.detach().numpy())
    return X_reconstructed

def pca_reduce(X, k):
    print("Performing PCA reduction ...")
    # Center the data
    X_centered = X - torch.mean(X, dim=0)

    # Compute the covariance matrix
    covariance = torch.mm(X_centered.t(), X_centered) / (X_centered.size(0) - 1)

    np_covariance = covariance.numpy().astype('float32')
    # np_covariance = sp.csr_matrix(np_covariance)
    vals, vecs = eigs(np_covariance, k=k, which='LR')
    vals = np.real(vals)
    vecs = np.real(vecs)
    sorted_indices = np.argsort(vals)
    # sorted_vals = vals[sorted_indices]
    sorted_vecs = vecs[:, sorted_indices]
    sorted_vecs = torch.from_numpy(sorted_vecs).float()

    # Project the centered data onto the selected eigenvectors
    X_reconstructed = torch.mm(X_centered, sorted_vecs).float()

    return X_reconstructed

def testNewdataloader():
    hhhh = ['bert_l-3_inference.json', 'bert_l-6_inference.json', 'bert_l-12_inference.json', 'resnet50_inference.json']
    hhhh = ['BlogCatalog', 'PT', 'ES', 'Flickr']
    hhhh = ['fe_rotor']
    for h in hhhh:
        print(h)
        # adj, _, _, _ = load_operator_data(h)
        # adj, _ = load_pyg_data(h)
        adj, _ = load_unweighted_partition_data(h)
        # adj = adj.tolil().todense()
        # print(adj.shape[0], ';', np.count_nonzero(adj))
        # usage='shortestPath', 'adjacentMat'
        X_reconstructed = get_pca_mat(h, adj, 20, usage='shortestPath')
        print(X_reconstructed.shape)
        X_reconstructed = get_pca_mat(h, adj, 20, usage='adjacentMat')
        print(X_reconstructed.shape)
# testNewdataloader()

def normalized_laplacian(adj):
    """
    Normalizes the given adjacency matrix using the degree matrix as
    \( \(D^{-1/2}(D-A)D^{-1/2}\) (symmetric normalization).
    """
    degrees = np.diagflat(np.sum(adj, axis=1))
    normalized_D = np.diagflat(np.power(np.sum(adj, axis=1), -0.5))
    # normalized_D = degree_power(adj, -0.5)
    output = normalized_D.dot((degrees - adj)).dot(normalized_D)
    
    return output

def normalized_adjacency(adj, symmetric=True):
    """
    Normalizes the given adjacency matrix using the degree matrix as either
    \(D^{-1}A\) or \(D^{-1/2}AD^{-1/2}\) (symmetric normalization).
    :param adj: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = degree_power(adj, -0.5)
        output = normalized_D.dot(adj).dot(normalized_D)
    else:
        normalized_D = degree_power(adj, -1.)
        output = normalized_D.dot(adj)
    return output


def degree_power(adj, pow):
    """
    Computes \(D^{p}\) from the given adjacency matrix. Useful for computing
    normalised Laplacians.
    :param adj: rank 2 array or sparse matrix
    :param pow: exponent to which elevate the degree matrix
    :return: the exponentiated degree matrix in sparse DIA format
    """
    degrees = np.diagflat(np.sum(adj, axis=1))
    normalized_D = np.diagflat(np.power(np.sum(adj, axis=1), pow))
    # degrees = np.power(np.array(adj.sum(1)), pow).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(adj):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return normalized_D
