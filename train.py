import time
import torch
import random
from datapro import CVEdgeDataset
from model import HRRLMDA, EmbeddingM, EmbeddingD, MDI
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret
def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric = caculate_metrics(y_pre, y_true)
    return metric
def caculate_metrics(pre_score, real_score):
    prc_result = []
    auc_result = []
    y_true = real_score
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)
    y_score = [0 if j < 0.5 else 1 for j in y_pre]
    prc_result.append(aupr)
    auc_result.append(auc)
    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)
    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)
    return metric_result
def get_metrics1(score, label):
    y_pre = score
    y_true = label
    metric, auc_result, prc_result, fpr, tpr, precision_u, recall_u = caculate_metrics1(y_pre, y_true)
    return metric, auc_result, prc_result, fpr, tpr, precision_u, recall_u
def caculate_metrics1(pre_score, real_score):
    prc_result = []
    auc_result = []
    y_true = real_score
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)
    y_score = [0 if j < 0.5 else 1 for j in y_pre]
    prc_result.append(aupr)
    auc_result.append(auc)
    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)
    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)
    return metric_result, auc_result, prc_result, fpr, tpr, precision_u, recall_u
def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))
def index(train_data, param,state):
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']
    kfolds = param.kfold
    torch.manual_seed(42)
    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)
        return train_edges, train_labels, test_edges, test_labels, train_idx, valid_idx
def Loader(i, simData, train_data, param,train_edges,train_labels, train_idx, valid_idx, state):
    kfolds = param.kfold
    model = HRRLMDA(EmbeddingM(param), EmbeddingD(param), MDI(param))
    model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    return model
def train_test(c, train_data, item1, simData, param, trainLoader, test_edges, test_labels, model):
        torch.manual_seed(42)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        running_loss = 0.0
        epo_label = []
        epo_score = []
        model.train()
        data, label = item1[c][0]
        c = c + 1
        train_data1 = data.cpu()
        true_label = label.cpu()
        pre_score, node_embed, total_loss = model(simData, train_data, train_data1, param)
        train_loss = torch.nn.BCELoss()
        pre_score = pre_score.float()
        true_label = true_label.float()
        score_loss = train_loss(pre_score, true_label)
        loss = score_loss+total_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        batch_score = pre_score.cpu().detach().numpy()
        epo_score = np.append(epo_score, batch_score)
        epo_label = np.append(epo_label, label.numpy())
        return pre_score, node_embed, loss, c
def train_test1(allneg_samples, train_data, param, simData, model):
    model.train()
    allneg_samples = torch.tensor(allneg_samples)
    pre_score, node_embed, total_loss = model(simData, train_data, allneg_samples, param)
    return pre_score, node_embed, total_loss






