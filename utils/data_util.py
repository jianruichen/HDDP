import os
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle
import pandas as pd
import scipy.sparse as sp
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin, mask_edges, mask_test_edges
from script.utils.make_edges_new import get_edges, get_prediction_edges, get_new_prediction_edges
import warnings
warnings.filterwarnings("ignore")

def load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=''):
    Adj = np.loadtxt(sim_path)
    Adj = sp.csr_matrix(Adj)
    return Adj

def load_protein_drug_interactions(threshold=0, toone=0, draw=0, path=''):
    protein_drug_interactions = np.loadtxt(path)
    drug_proten_interactions = protein_drug_interactions.transpose()
    # Sparse process.....
    drug_proten_interactions = sp.csr_matrix(drug_proten_interactions)
    return drug_proten_interactions

def load_last_drug_protein(a):
    drug_protein = a
    drug_protein_high = pd.read_excel('E:/硕士/第二篇-JOU/drug-disease/CODE/HTGN-main/data/B-dataset/fo_order_dr_pro.xlsx', header=None)
    high = drug_protein + drug_protein_high
    # # # df = pd.DataFrame(high)
    # # # df.to_excel('E:/硕士/第二篇-JOU/drug-disease/CODE/HTGN-main/data/drug-1/last_drug_protein.xlsx', header=None, index=False)
    # high[high >= 1] = 1
    return high

def load_last_protein_disease(a):
    protein_disease = a
    protein_disease_high = pd.read_excel('E:/硕士/第二篇-JOU/drug-disease/CODE/HTGN-main/data/B-dataset/fo_order_pro_dis.xlsx', header=None)
    high = protein_disease + protein_disease_high
    # # df = pd.DataFrame(high)
    # # df.to_excel('E:/硕士/第二篇-JOU/drug-disease/CODE/HTGN-main/data/drug-1/last_protein_disease.xlsx', header=None, index=False)
    # high[high >= 1] = 1
    return high

def load_last_drug_disease(a):
    protein_disease = a.toarray()
    # protein_disease_high = pd.read_excel('E:/硕士/第二篇-JOU/drug-disease/CODE/HTGN-main/data/drug-1/PDhigher.xlsx', header=None)
    # high = protein_disease + protein_disease_high
    # df = pd.DataFrame(high)
    # df.to_excel('E:/硕士/第二篇-JOU/drug-disease/CODE/HTGN-main/data/drug-1/last_protein_disease.xlsx', header=None, index=False)
    # protein_disease[protein_disease >= 1] = 1
    return protein_disease

def load_data_drug_disease(data_path):
    protein_drug_path = os.path.join(data_path, "mat_protein_drug.txt")
    protein_disease_path = os.path.join(data_path, "mat_protein_disease.txt")
    drug_disease_path = os.path.join(data_path, "mat_drug_disease.txt")

    global drug_proten_interactions, protein_disease_adj, drug_disease_adj
    drug_proten_interactions = load_protein_drug_interactions(path=protein_drug_path)
    protein_disease_adj = load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=protein_disease_path)
    drug_disease_adj = load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=drug_disease_path)

    return drug_proten_interactions.toarray(), protein_disease_adj.toarray(), drug_disease_adj.toarray()

def load_data_dataset(data_path):
    u = pd.read_csv("E:\\硕士\\第一篇\\DsGCN\\data\\ncrna\\l-d.csv")
    m = u.shape[0]  #c-d:6437  l-d:139329
    number_rna = int(max(u["RNA_id"]))  # 212
    number_disease = int(max(u["dise_id"]))  # 90
    N = number_rna + number_disease  # 302
    score_matrix = np.zeros([number_rna, number_disease])  # (212,90)
    for i in range(0, m):
        if u["score"][i] > 0:  # circ 829 lncrna 23
            score_matrix[int(u["RNA_id"][i] - 1)][int(u["dise_id"][i] - 1)] = 1
        else:
            score_matrix[int(u["RNA_id"][i] - 1)][int(u["dise_id"][i] - 1)] = 0
    # df = pd.DataFrame(score_matrix)
    # df.to_excel('E:\\硕士\\第一篇\\HML\\data\\ncrna\\lnc_disease.xlsx', header=None, index=False)
    return score_matrix

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_vgrnn_dataset(dataset):
    data_path = os.path.join('E:/硕士/第一篇/HML/data', dataset)
    rna_disease_adj = load_data_dataset(data_path)
    adj_rna_disease = rna_disease_adj
    adjT = np.transpose(adj_rna_disease)
    first = np.zeros((adj_rna_disease.shape[0], adj_rna_disease.shape[0]))
    second = np.zeros((adj_rna_disease.shape[1], adj_rna_disease.shape[1]))
    first_1 = np.hstack((first, adj_rna_disease))
    second_1 = np.hstack((adjT, second))
    adj_matrix = np.vstack((first_1, second_1))
    adj = sp.csr_matrix(adj_matrix)
    data = {}
    edge_index_list, pedges_list, nedges_list = mask_edges(adj, test_prop=0.2)

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    # data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = adj.shape[0]
    data['adj'] = adj_matrix
    # data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    # print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data

def load_dataset(dataset):
    data_train = pd.read_csv(
        "E:\\硕士\\第一篇\\HML\\data\\didr\\dige.csv", header=None,  # 读取的CSV文件没有列名，因此在读取过程中使用header=None参数指定不使用列名
        names=['d_nodes', 'g_nodes', 'relations'], sep='\t')  # 读出的形式为[0  DB12582  1437  -1.0]..., dtype=dtypes参数可以指定每列的数据类型

    data_array_train = data_train.values.tolist()  # 转换为列表，形式为[['DB12582'  '1437'  '-1.0'],['DB12582'  '7402'  '1.0']...]
    data_array_train = np.array(data_array_train)  # 将列表转换为一个NumPy数组{ndarray:(64582,3)}

    d_nodes_relations = data_array_train[:, 0]  # 存放药物名那一列的数据
    g_nodes_relations = data_array_train[:, 1]
    #ddi
    # data_array = np.concatenate([d_nodes_relations, g_nodes_relations],
    #                             axis=0)  # 转化为np.array进行np.concatenate data_array也是np.array类型
    # data = list(set(data_array))
    # dictionary = {value: i for i, value in enumerate(data)}
    #ddil
    data_0 = list(set(d_nodes_relations))
    data_1 = list(set(g_nodes_relations))

    dictionary_0 = {value: i for i, value in enumerate(data_0)}
    dictionary_1 = {value: j for j, value in enumerate(data_1)}

    data_0 = np.array([dictionary_0[x] for x in d_nodes_relations])
    data_1 = np.array([dictionary_1[y] for y in g_nodes_relations])

    d_nodes = data_0  # 药物索引列
    g_nodes = data_1  # 基因索引列

    # labels是将一个以d为行，g为列的数组reshape为一维数组
    pairs_nonzero = np.array([[d, g] for d, g in zip(d_nodes, g_nodes)])  # 80000行中的d，g对,把d、g结合
    m = pairs_nonzero.shape[0]
    number_rna = int(max(d_nodes))
    number_disease = int(max(g_nodes))
    # D = max(number_rna,number_disease) ddi
    # score_matrix = np.zeros([D, D])
    score_matrix = np.zeros([number_rna, number_disease])
    for i in range(0, m):
        a = pairs_nonzero[i][0]-1
        b = pairs_nonzero[i][1]-1
        score_matrix[a][b] = 1
    rna_disease_adj = score_matrix
    adj_rna_disease = rna_disease_adj
    adjT = np.transpose(adj_rna_disease)
    first = np.zeros((adj_rna_disease.shape[0], adj_rna_disease.shape[0]))
    second = np.zeros((adj_rna_disease.shape[1], adj_rna_disease.shape[1]))
    first_1 = np.hstack((first, adj_rna_disease))
    second_1 = np.hstack((adjT, second))
    adj_matrix = np.vstack((first_1, second_1))
    adj = sp.csr_matrix(adj_matrix)
    data = {}
    edge_index_list, pedges_list, nedges_list = mask_edges(adj, test_prop=0.2)

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    # data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = adj.shape[0]
    data['adj'] = adj_matrix
    # data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    # print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
    pedges, nedges = get_prediction_edges(undirected_edges)  # list
    new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

    data['edge_index_list'] = undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length: {}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_vgrnn_dataset_det(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset_det(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1

    gdata_list = []
    for edge_index in undirected_edges:
        gdata = Data(x=None, edge_index=edge_index, num_nodes=num_nodes)
        gdata_list.append(train_test_split_edges(gdata, 0.1, 0.4))

    data['gdata'] = gdata_list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length: {}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def loader(dataset='enron10'):
    # if cached, load directly
    data_root = '../data/'.format(dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)
    if os.path.isfile(filepath):
        print('loading {} directly'.format(dataset))
        # return torch.load(filepath)
    # if not cached, to process and cached
    # print('>>data is not exits, processing ...')
    if dataset in ['enron10', 'dblp', 'drug-1', 'B-dataset', 'F-dataset', 'm-dataset', 'ncrna']:
        data = load_vgrnn_dataset(dataset)
    if dataset in ['as733', 'fbw', 'HepPh30', 'disease']:
        data = load_new_dataset(dataset)
    if dataset in ['didr']:
        data = load_dataset(dataset)
    # torch.save(data, filepath)
    # print('saved!')
    return data
