import numpy as np
import scipy.sparse as sp
import torch


def mask_test_edges(adj):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:  检查对角线元素是否为0
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # triu 取出稀疏矩阵的上三角部分的非零元素 元素值及其坐标
    adj_tuple = sparse_to_tuple(adj_triu)  # 将上三角转为tuple 返回三元组形式
    edges = adj_tuple[0]  # 返回坐标值 coords
    # train_edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]  # 这是从整个邻接矩阵得到的边的坐标
    num_test = int(np.floor(edges.shape[0] * 0.2)) # 20%数量的边作为测试集

    all_edge_idx = list(range(edges.shape[0]))  # edges应该是一个两位数组 每一行是一个坐标 列数就是所有边的总个数
    np.random.shuffle(all_edge_idx)  # 通过打乱索引 来进行shuffle 而不是直接shuffle原数据
    test_edge_idx = all_edge_idx[:num_test]   # 测试集边的索引
    test_edges = edges[test_edge_idx]   # 通过索引指定对应的测试集的边 313398
    train_edges = np.delete(edges, np.hstack([test_edge_idx]), axis=0)  # 把test删掉就是训练集的边 1253594
    ### ！！！ 注意 因为adj确认了没有0 所以所有的test 和train edge都是正例！

    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)
    m_pos = test_edges.shape[0]  # 313398
    test_edges_false = neg_edges[:m_pos]

    # def ismember(a, b, tol=5):
    #     rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    #     return np.any(rows_close)

    # ～是取反的意思
    # 以下五句话分别是确认：
    # 为测试集生成的负样本的边坐标不在所有正样本边集合里面
    # 测试集正样本不在训练集里面
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(test_edges, train_edges)
    # NOTE: these edge lists only contain single direction of edge!
    return train_edges, test_edges, test_edges_false


def mask_edges(adj, test_prop):
    adj[adj >= 1] = 1  # get tp edges
    x, y = sp.triu(adj).nonzero()  #取出稀疏矩阵上三角部分的元素
    pos_edges = np.array(list(zip(x, y)))  #(a,b) 是从a，b这两个序列中，成对的、按顺序的取出相同位置上的元素，然后放在一起，组成一个元组
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)  #1566992
    # n_val = int(m_pos * test_prop)  #156699
    n_test = int(m_pos * test_prop)  #313398
    test_edges, train_edges = pos_edges[:n_test], pos_edges[n_test:]
    test_edges_false = neg_edges[:n_test]
    return train_edges, torch.LongTensor(test_edges), torch.LongTensor(test_edges_false)

def mask_edges_prd(adjs_list):
    pos_edges_l, false_edges_l = [], []
    edges_list = []
    # for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    adj = adjs_list
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_false = int(edges.shape[0])

    pos_edges_l.append(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edges_false = []
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    assert ~ismember(edges_false, edges_all)

    false_edges_l.append(np.asarray(edges_false))

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def test_adj(adjs_list, adj_orig_dense_list):
    # this method is to test the adj_list and adj_orig_dense_list is same or not
    for i, a in enumerate(adj_orig_dense_list):
        a = sp.csr_matrix(a)
        a = a - sp.dia_matrix((a.diagonal()[np.newaxis, :], [0]), shape=a.shape)
        a.eliminate_zeros()
        assert np.diag(a.todense()).sum() == 0
        team1 = sp.csr_matrix(a).todok().tocoo()
        print(len(list(team1.col.reshape(-1))))

        adj = adjs_list[i]
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        assert np.diag(adj.todense()).sum() == 0
        team2 = sp.csr_matrix(adj).todok().tocoo()

        print(len(list(team2.col.reshape(-1))))
        print('==')


def mask_edges_prd_new(adjs_list, adj_orig_dense_list):
    # produce new edge index
    pos_edges_l, false_edges_l = [], []

    # the first snapshots
    adj = adjs_list[0]
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]  # single direction
    edges_all = sparse_to_tuple(adj)[0]  # all

    pos_edges_l.append(edges)

    num_false = int(edges.shape[0])

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edges_false = []
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    assert ~ismember(edges_false, edges_all)
    false_edges_l.append(np.asarray(edges_false))

    # the next snapshots
    for i in range(1, len(adjs_list)):
        edges_pos = np.transpose(np.asarray(np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i - 1]) > 0)))
        num_false = int(edges_pos.shape[0])
        adj = adjs_list[i]  # current adj
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        assert np.diag(adj.todense()).sum() == 0

        edges_all = sparse_to_tuple(adj)[0]

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:  # filter self-loop
                continue
            if ismember([idx_i, idx_j], edges_all):  # filter old edges
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)

        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def mask_edges_prd_new_by_marlin(adjs_list):
    # This code is same with the previous one but only need to one spare adj matrix
    pos_edges_l, false_edges_l = [], []

    # 1. the first snapshots
    adj = adjs_list
    # 1.1 Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # 1.2 Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]  # single direction
    edges_all = sparse_to_tuple(adj)[0]  # all

    pos_edges_l.append(edges)

    num_false = int(edges.shape[0])

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    # 1.3 negative sampling
    edges_false = []
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    assert ~ismember(edges_false, edges_all)
    false_edges_l.append(np.asarray(edges_false))

    # # 2. the next snapshots
    # for i in range(1, len(adjs_list)):
    #     # 2.1 get new edge_index
    #     edges = sparse_to_tuple(adjs_list[i])[0]  # current edges
    #     last_edges = sparse_to_tuple(adjs_list[i - 1])[0]  # last edges
    #     edges_perm = edges[:, 0] * 1e5 + edges[:, 1]  # hash current edges
    #     last_edges_perm = last_edges[:, 0] * 1e5 + last_edges[:, 1]  # hash last edges
    #     perm = np.setdiff1d(edges_perm, np.intersect1d(edges_perm, last_edges_perm))  # new edges: edge-edge^last_edge
    #     edges_pos = np.vstack(np.divmod(perm, 1e5)).transpose().astype(np.long)  # convert perm to indices
    #     num_false = int(edges_pos.shape[0])
    #
    #     # 2.2 get all pos edge to avoid being sampled
    #     adj = adjs_list[i]  # current adj
    #     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    #     adj.eliminate_zeros()
    #     assert np.diag(adj.todense()).sum() == 0
    #     edges_all = sparse_to_tuple(adj)[0]
    #
    #     # 2.3 sample equal size of neg edges
    #     edges_false = []
    #     while len(edges_false) < num_false:
    #         idx_i = np.random.randint(0, adj.shape[0])
    #         idx_j = np.random.randint(0, adj.shape[0])
    #         if idx_i == idx_j:  # filter self-loop
    #             continue
    #         if ismember([idx_i, idx_j], edges_all):  # filter old edges
    #             continue
    #         if edges_false:
    #             if ismember([idx_j, idx_i], np.array(edges_false)):
    #                 continue
    #             if ismember([idx_i, idx_j], np.array(edges_false)):
    #                 continue
    #         edges_false.append([idx_i, idx_j])
    #
    #     assert ~ismember(edges_false, edges_all)
    #
    #     false_edges_l.append(np.asarray(edges_false))
    #     pos_edges_l.append(edges_pos)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    return out


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_edges_det(adjs_list):
    '''
    produce edge_index in np format
    '''
    edges_list = []
    biedges_list = []
    # for i in range(0, len(adjs_list)):
    adj = adjs_list
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]  # single directional
    np.random.shuffle(edges)
    edges_list.append(edges)
    biedges = sparse_to_tuple(adj)[0]  # bidirectional edges
    np.random.shuffle(biedges)
    biedges_list.append(biedges)

    return edges_list, biedges_list
