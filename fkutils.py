import numpy as np
import sys, time, os
import networkx as nx
import scipy.sparse as sp


def findResDuringAllExperi(P, n_clust, res_list, edge_index, node_feature):
    num_experi = len(res_list)
    lc_list = []
    lo_list = []
    S_list = []

    tl_list = []
    tl_lc_list = []
    tl_lo_list = []

    for idx in range(num_experi):
        lc, lo, tl, best_S = res_list[idx]
        # print(lc, lo, tl)
        lc_list.append(lc)
        lo_list.append(lo)
        tl_list.append(tl[0]+tl[1])
        tl_lc_list.append(tl[0])
        tl_lo_list.append(tl[1])
        S_list.append(best_S)
    

    A = edge_index
    X = node_feature
    

    best_lc = min(lc_list)
    best_lc_idx = lc_list.index(best_lc)
    _, lo_b_lc, _ = calLoss(S_list[best_lc_idx][1], X, A, n_clust)
    tl_b_lc = best_lc + lo_b_lc
    cc_b_lc = compute_communication_cost(A, S_list[best_lc_idx][1])
    ps_b_lc = max(compute_each_partition_size(X, S_list[best_lc_idx][1], P['bigG']))
    imb_b_lc = ps_b_lc/np.ceil(np.sum(X)/n_clust) - 1

    best_lo = min(lo_list)
    best_lo_idx = lo_list.index(best_lo)
    lc_b_lo, _, _ = calLoss(S_list[best_lo_idx][2], X, A, n_clust)
    tl_b_lo = best_lo + lc_b_lo
    cc_b_lo = compute_communication_cost(A, S_list[best_lo_idx][2])
    ps_b_lo = max(compute_each_partition_size(X, S_list[best_lo_idx][2], P['bigG']))
    imb_b_lo = ps_b_lo/np.ceil(np.sum(X)/n_clust) - 1

    best_tl = min(tl_list)
    best_tl_idx = tl_list.index(best_tl)
    lc_b_tl = tl_lc_list[best_tl_idx]
    lo_b_tl = tl_lo_list[best_tl_idx]
    cc_b_tl = compute_communication_cost(A, S_list[best_tl_idx][0])
    ps_b_tl = max(compute_each_partition_size(X, S_list[best_tl_idx][0], P['bigG']))
    imb_b_tl = ps_b_tl/np.ceil(np.sum(X)/n_clust) - 1

    mean_tl = sum(tl_list)/num_experi
    mean_lc = sum(lc_list)/num_experi
    mean_lo = sum(lo_list)/num_experi
    print(P['dataset'], " " + str(n_clust) + " partitions")
    print("best_tl", ":", best_tl)
    print("lc_b_tl", ":", lc_b_tl)
    print("lo_b_tl", ":", lo_b_tl)

    return [best_tl, lc_b_tl, lo_b_tl, cc_b_tl, ps_b_tl, imb_b_tl,
            best_lc, lo_b_lc, tl_b_lc, cc_b_lc, ps_b_lc, imb_b_lc,
            best_lo, lc_b_lo, tl_b_lo, cc_b_lo, ps_b_lo, imb_b_lo,
            mean_lc, mean_lo, mean_tl], \
            [S_list[best_lc_idx][1], S_list[best_lo_idx][2], S_list[best_tl_idx][0]], best_tl_idx

def compute_degree_mat_of_adjM(A):
    reduce_sum = np.add.reduce(A, 0).reshape((1, A.shape[0]))
    degree_mat = np.zeros(shape=(A.shape[0], A.shape[0]))
    for i in range(reduce_sum.shape[-1]):
        degree_mat[i, i] = reduce_sum[0, i]
    return degree_mat

def GON_Metrics(resS, list_X, A_in):
    
    meansize = np.sum(list_X)/resS.shape[-1]
    
    clusters = list(np.count_nonzero(resS, axis=0))
    bci = 0.
    for clu in clusters:
        bci += np.abs(clu-meansize)/meansize
    bci /= resS.shape[-1]
    
    if sp.issparse(A_in):
        resS = sp.csr_matrix(resS)
        resS_t = resS.transpose()
    A_pooled = (resS_t.dot(A_in)).dot(resS)
    gin = 0.5*A_pooled.trace()
    
    return gin, bci

def calLoss(resS, list_X, A_in, n_clust):
    if sp.issparse(A_in):
        resS = sp.csr_matrix(resS)
        resS_t = resS.transpose()
        A_pooled = (resS_t.dot(A_in)).dot(resS)
        degree = A_in.sum(axis=0)
        degrees = degree.tolist()[0]
        D = sp.diags(degrees)
        tmpm = (resS_t @ D) @ resS
        safety = sp.csr_matrix(np.diagflat([1e-3 for i in range(tmpm.shape[0])]))
        tmpm = tmpm + safety
        Ap_mul_tmpm = A_pooled @ sp.linalg.inv(tmpm)
        cut_loss = -Ap_mul_tmpm.trace() / n_clust
        # list_X = np.squeeze(list_X)#.tolist()[0]
        list_X = np.reshape(list_X, (A_in.shape[0],)).tolist()

        degree_mat = sp.diags(list_X)
        list_X = degree_mat
        SXS = (resS_t @ list_X) @ resS
        mean_X = np.sum(list_X) / n_clust
        multipilers = 1. / ((n_clust - 1) * mean_X)
        I_S = np.eye(n_clust)
        lo_mat = multipilers * (SXS - mean_X * I_S)
        ortho_loss = np.linalg.norm(lo_mat)
    else:
        A_pooled = np.matmul(np.matmul(np.transpose(resS), A_in), resS)
        D = compute_degree_mat_of_adjM(A_in)
        tmpm = np.matmul(np.matmul(np.transpose(resS), D), resS)
        Ap_mul_tmpm = np.matmul(A_pooled, np.linalg.inv(tmpm))
        cut_loss = -np.trace(Ap_mul_tmpm) / n_clust

        degree_mat = np.zeros(shape=(A_in.shape[0], A_in.shape[0]))
        for i in range(A_in.shape[0]):
            degree_mat[i][i] = list_X[i, 0]
        list_X = degree_mat
        SXS = np.matmul(np.matmul(np.transpose(resS), list_X), resS)
        mean_X = np.sum(list_X) / n_clust
        multipilers = 1. / ((n_clust - 1) * mean_X)
        I_S = np.eye(n_clust)
        lo_mat = multipilers * (SXS - mean_X * I_S)
        ortho_loss = np.linalg.norm(lo_mat)
    # print(cut_loss, ortho_loss, cut_loss + ortho_loss)
    return cut_loss, ortho_loss, cut_loss + ortho_loss

def getPartionedGraph(sp_adj, assign, method, isweighted, dataset):
    logpath = '/home/pengcheng/logs/e2egplogs/'
    par1adjPath = logpath + 'downtasks/{}-{}-{}-par1_adj.npz'.format(dataset, method, isweighted)
    par1uniidPath = logpath + 'downtasks/{}-{}-{}-par1_uniid.npz'.format(dataset, method, isweighted)
    par2adjPath = logpath + 'downtasks/{}-{}-{}-par2_adj.npz'.format(dataset, method, isweighted)
    par2uniidPath = logpath + 'downtasks/{}-{}-{}-par2_uniid.npz'.format(dataset, method, isweighted)
    # if False:
    if os.path.exists(par1uniidPath):
        par1 = sp.load_npz(par1adjPath).astype('float32')
        uniq_ids1 = np.load(par1uniidPath)
        par2 = sp.load_npz(par2adjPath).astype('float32')
        uniq_ids2 = np.load(par2uniidPath)
        tup_g1 = (par1, uniq_ids1)
        tup_g2 = (par2, uniq_ids2)
    else:
        sp_adj = sp.coo_matrix(sp_adj)
        part1 = []
        part2 = []
        for i,j,v in zip(sp_adj.row, sp_adj.col, sp_adj.data):
            if np.nonzero(assign[i]) == np.nonzero(assign[j]):
                part_id = np.nonzero(assign[i])[0]
                if part_id == 0:
                    part1.append([i,j,float(v)])
                elif part_id == 1:
                    part2.append([i,j,float(v)])

        par1_arr = np.asarray(part1)
        par2_arr = np.asarray(part2)
        tup_g1 = oderEdgeArr(par1_arr[:, 0], par1_arr[:, 1], par1_arr[:, 2])
        tup_g2 = oderEdgeArr(par2_arr[:, 0], par2_arr[:, 1], par2_arr[:, 2])
        sp.save_npz(par1adjPath, tup_g1[0])
        sp.save_npz(par2adjPath, tup_g2[0])
        np.savez(par1uniidPath, tup_g1[1])
        np.savez(par2uniidPath, tup_g2[1])
    return tup_g1, tup_g2

def getNxGraphfromAdj(par1, par2):
    g1 = nx.from_scipy_sparse_array(par1)
    g2 = nx.from_scipy_sparse_array(par2)
    return g1, g2

def SVDonPars(par1, par2, k=10):
    time1 = time.time()
    sp.linalg.eigs(par1, k=k, which='SR')
    time1 = time.time() - time1
    
    time2 = time.time()
    sp.linalg.eigs(par2, k=k, which='SR')
    time2 = time.time() - time2
    return max(time1, time2)


def louvainonPars(g1, g2):
    # print(g1.number_of_nodes(), g2.number_of_nodes())
    time1 = time.time()
    # pr = nx.dfs_edges(g1)
    # pr = nx.pagerank(g1)
    pr1 = nx.community.louvain_communities(g1, threshold=1)
    time1 = time.time() - time1

    time2 = time.time()
    # pr = nx.dfs_edges(g2)
    # pr = nx.pagerank(g2)
    pr2 = nx.community.louvain_communities(g2, threshold=1)
    time2 = time.time() - time2
    return max(time1, time2), pr1, pr2

def pageRankonPars(g1, g2):
    # print(g1.number_of_nodes(), g2.number_of_nodes())
    time1 = time.time()
    # pr = nx.dfs_edges(g1)
    pr = nx.pagerank(g1)
    time1 = time.time() - time1

    time2 = time.time()
    # pr = nx.dfs_edges(g2)
    pr = nx.pagerank(g2)
    time2 = time.time() - time2
    return max(time1, time2)

def oderEdgeArr(row_arr, col_arr, weights):
    row_lst = []
    col_lst = []
    uniq_ids = np.unique(np.append(row_arr, col_arr))
    for idx in range(row_arr.shape[0]):
        row = np.searchsorted(uniq_ids, row_arr[idx])
        col = np.searchsorted(uniq_ids, col_arr[idx])
        row_lst.append(row)
        col_lst.append(col)

    row_arr = np.array(row_lst)
    col_arr = np.array(col_lst)
    weights = np.array(weights)

    adj_sp = sp.csr_matrix((weights, (row_arr, col_arr)))
    
    return (adj_sp, uniq_ids)

# find the best Lc, best Lo, best Total loss and their idx
def findBestResDuringTraining(P, n_clust, loss_list, S_list):
    best_loss = 100
    lc_in_best_loss = 100
    lo_in_best_loss = 100
    best_loss_idx = 100
    best_lc = 0.
    best_lc_idx = 100
    idea_lc = 0.
    idea_lc_idx = 100
    best_lo = 100
    best_lo_idx = 100
    len_loss = len(loss_list)
    for idx in range(len_loss):
        lc = loss_list[idx][0]
        lo = loss_list[idx][1]
        total_loss = lc + lo
        if lc < best_lc:
            best_lc = lc
            best_lc_idx = idx
        if lo < best_lo:
            best_lo = lo
            best_lo_idx = idx
        if total_loss < best_loss:
            best_loss = total_loss
            lc_in_best_loss = lc
            lo_in_best_loss = lo
            best_loss_idx = idx
    if idea_lc != 0.:
        best_lc = idea_lc
        best_lc_idx = idea_lc_idx
    best_S = S_list[best_loss_idx]
    best_lc_S = S_list[best_lc_idx]
    best_lo_S = S_list[best_lo_idx]
    return [best_lc, best_lo, [lc_in_best_loss, lo_in_best_loss], [best_S, best_lc_S, best_lo_S]]


def get_node_imb(X, assignment):
    k = assignment.shape[1]
    maxp = max(compute_each_partition_size(X, assignment, issparse=True))
    imb = maxp/np.ceil(np.sum(X)/k) - 1
    return imb

def get_edge_imb(adjacency_matrix, part_matrix):
    A_in = adjacency_matrix
    resS = part_matrix
    resS = sp.csr_matrix(resS)
    resS_t = resS.transpose()
    A_pooled = (resS_t.dot(A_in)).dot(resS)
    mean = np.sum(A_in)/part_matrix.shape[1]
    return np.max(A_pooled.diagonal())/mean - 1
    
def compute_communication_cost(adjacency_matrix, part_matrix):
    if sp.issparse(adjacency_matrix):
        A_in = adjacency_matrix
        resS = part_matrix
        resS = sp.csr_matrix(resS)
        resS_t = resS.transpose()
        A_pooled = (resS_t.dot(A_in)).dot(resS)
        edge_cut = (np.sum(A_in) - np.sum(A_pooled.diagonal()))/2
    else:
        edge_cut = 0.0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i, adjacency_matrix.shape[0]):
                if adjacency_matrix[i, j] != 0:
                    # print(part_matrix[j])
                    # print(part_matrix[i])
                    if np.nonzero(part_matrix[j]) != np.nonzero(part_matrix[i]):
                        edge_cut = edge_cut + adjacency_matrix[i, j]
    return edge_cut

def soft2hard(assignment):
    # find the maximum value in each row
    max_values = np.amax(assignment, axis=1)

    # create a new matrix with the same shape as the original matrix
    hard = np.zeros_like(assignment)

    # set the max value in each row to 1 and the rest to 0
    for i in range(assignment.shape[0]):
        hard[i] = np.where(assignment[i] == max_values[i], 1, 0)
    return hard


def compute_each_partition_size(node_feature, cluster_matrix, issparse=False):
    cluster_num = len(cluster_matrix[0])
    nodes_num = len(node_feature)
    partition_size_list = []
    if issparse:
        resS = sp.csr_matrix(cluster_matrix) 
        resS_t = resS.transpose()
        list_X = node_feature
        list_X = np.squeeze(list_X).tolist()
        degree_mat = sp.diags(list_X)
        list_X = degree_mat
        SXS = (resS_t @ list_X) @ resS
        for i in range(cluster_num):
            partition_size_list.append(SXS[i, i])

    else:
        feature_correct_matrix = np.zeros(shape=(nodes_num, nodes_num), dtype=float)
        for i in range(nodes_num):
            feature_correct_matrix[i, i] = node_feature[i, 0]
        StXS = np.matmul(np.matmul(np.transpose(cluster_matrix), feature_correct_matrix), cluster_matrix)
        
        for i in range(cluster_num):
            partition_size_list.append(StXS[i, i])
    return partition_size_list

