import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
import random
import os
import torch
import csv
import torch.utils.data.dataset as Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist

def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data



def loading_data(param):
    ratio = param.ratio
    md_matrix = np.loadtxt(os.path.join(param.datapath + '/m_d.csv'), dtype=int, delimiter=',')
    md_matrix = torch.tensor(md_matrix)
    zero_index = []
    one_index = []
    for i in range(md_matrix.size(0)):
        for j in range(md_matrix.size(1)):
            if md_matrix[i][j] < 1:
                zero_index.append([i, j])
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])

    rng = np.random.default_rng(seed=42)
    pos_samples = np.where(md_matrix == 1)
    pos_samples_shuffled = rng.permutation(pos_samples)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    one_index = np.array(one_index)
    zero_index = np.array(zero_index)
    zero_index = zero_index.T
    neg_samples = np.where(md_matrix == 0)
    allneg_samples = rng.permutation(neg_samples)
    allneg_samples = allneg_samples.T
    neg_samples_shuffled = rng.permutation(zero_index)[:, :pos_samples_shuffled.shape[1]]
    neg_sample= neg_samples_shuffled.T
    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]
    idx_split = int(n_pos_samples * ratio)
    one_index = one_index.T
    one_index1 = one_index.T
    sample = np.vstack((one_index1, neg_sample))
    test_pos_edges = one_index[:, :idx_split]
    test_neg_edges = neg_samples_shuffled[:, :idx_split]
    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))
    test_edges, test_true_label = shuffle(test_edges, test_true_label, random_state=42)
    train_pos_edges = one_index[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
    train_edges, train_true_label = shuffle(train_edges, train_true_label, random_state=42)
    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label
    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label
    edge_idx_dict['true_md'] = md_matrix  ##*
    true_label = np.hstack((np.ones(one_index1.shape[0]), np.zeros(neg_sample.shape[0])))
    kfolds = param.kfold
    torch.manual_seed(42)
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    train_dex, valid_dex = [], []
    for train_index, valid_index in kf.split(sample):
        train_dex.append(train_index)
        valid_dex.append(valid_index)
    return train_dex, valid_dex, sample, true_label, edge_idx_dict, allneg_samples
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)
# 汉明距离计算函数
def hamming_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sum(x != y)
# 行行之间的相似性
def compute_row_similarity(data):
    n_rows = data.shape[0]
    row_distances = np.zeros((n_rows, n_rows))
    # 计算每一对行之间的汉明距离
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            distance = hamming_distance(data[i], data[j])
            row_distances[i, j] = distance
            row_distances[j, i] = distance
    # 转换为相似性矩阵 (1 / (1 + 距离))
    row_similarity = 1 / (1 + row_distances)
    return row_similarity
# 列列之间的相似性
def compute_column_similarity(data):
    n_cols = data.shape[1]
    col_distances = np.zeros((n_cols, n_cols))
    # 计算每一对列之间的汉明距离
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            distance = hamming_distance(data[:, i], data[:, j])
            col_distances[i, j] = distance
            col_distances[j, i] = distance
    # 转换为相似性矩阵 (1 / (1 + 距离))
    col_similarity = 1 / (1 + col_distances)
    return col_similarity
# 计算行和列之间的相似性
def compute_hamming_similarity(data):
    # 行行之间的相似性
    row_similarity = compute_row_similarity(data)
    row_similarity = row_similarity.astype(np.float32)
    # 列列之间的相似性
    col_similarity = compute_column_similarity(data)
    col_similarity = col_similarity.astype(np.float32)
    return row_similarity, col_similarity
def Simdata_pro(param):
    dataset = dict()
    "miRNA-disease association"
    m_d_matrix = read_csv(param.datapath + '/m_d.csv')
    row_similarity, col_similarity = compute_hamming_similarity(m_d_matrix)

    "miRNA sequence sim"
    mm_s_matrix = read_csv(param.datapath + '/m_ss.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    "disease target-based sim"
    dd_t_matrix = read_csv(param.datapath + '/d_ts.csv')
    dd_t_edge_index = get_edge_index(dd_t_matrix)
    dataset['dd_t'] = {'data_matrix': dd_t_matrix, 'edges': dd_t_edge_index}

    "miRNA functional sim"
    mm_f_matrix = read_csv(param.datapath + '/m_fs.csv')
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}

    "disease semantic sim"
    dd_s_matrix = read_csv(param.datapath + '/d_ss.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    "miRNA hamming sim"
    # mm_g_matrix = read_csv(param.datapath + '/m_gs.csv')
    mm_g_matrix = torch.from_numpy(row_similarity)
    mm_g_edge_index = get_edge_index(mm_g_matrix)
    dataset['mm_g'] = {'data_matrix': mm_g_matrix, 'edges': mm_g_edge_index}

    "disease hamming sim"
    # dd_g_matrix = read_csv(param.datapath + '/d_gs.csv')
    dd_g_matrix = torch.from_numpy(col_similarity)
    dd_g_edge_index = get_edge_index(dd_g_matrix)
    dataset['dd_g'] = {'data_matrix': dd_g_matrix, 'edges': dd_g_edge_index}

    return dataset
class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):

        self.Data = edges
        self.Label = labels
    def __len__(self):
        return len(self.Label)
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label




