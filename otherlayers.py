from torch import nn as nn
import torch
from math import sqrt
class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name
    def forward(self, x):
        return self.bn(x)
class BnodeEmbedding(nn.Module):
    def __init__(self, embedding, dropout, freeze=False):
        super(BnodeEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(embedding, dtype=torch.float32).detach(), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout
    def forward(self, x):
        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x
class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
    def forward(self, x, node1):
        model1 = MyModel2(node1)
        x = model1(x)#batchsize*featuresize
        if self.outBn: x = self.bns(x) if len(x.shape) == 2 else self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x
class GCN(nn.Module):
    def __init__(self, inSize, outSize, dropout, layers, resnet, actFunc, outBn=False, outAct=True, outDp=True):
        super(GCN, self).__init__()
        self.gcnlayers = layers
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet
    def forward(self, x, L, node1):
        Z_zero = x
        m_all = Z_zero[:, 0, :].unsqueeze(dim=1)
        d_all = Z_zero[:, 1, :].unsqueeze(dim=1)
        for i in range(self.gcnlayers):
            a = (torch.matmul(L, x))
            model = MyModel(node1)
            a = model(a)
            if self.outBn:
                if len(L.shape) == 3:
                    a = self.bns(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = self.bns(a)
            if self.outAct: a = self.actFunc(a)
            if self.outDp: a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a
            m_this = x[:, 0, :].unsqueeze(dim=1)
            d_this = x[:, 1, :].unsqueeze(dim=1)
            m_all = torch.cat((m_all, m_this), 1)
            d_all = torch.cat((d_all, d_this), 1)
        return m_all, d_all
class MyModel(nn.Module):
    def __init__(self, node1):
        super(MyModel, self).__init__()
        self.out = nn.Linear(in_features=node1, out_features=node1, bias=True)
    def forward(self, x):
        return self.out(x)
class LayerAtt(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers):
        super(LayerAtt, self).__init__()
        self.layers = gcnlayers + 1
        self.inSize = inSize
        self.outSize = outSize
        self.q = nn.Linear(inSize, outSize)
        self.k = nn.Linear(inSize, outSize)
        self.v = nn.Linear(inSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=1)
        self.actfun2 = nn.ReLU()
        self.attcnn = nn.Conv1d(in_channels=self.layers, out_channels=1, kernel_size=1, stride=1,
                            bias=True)
    def forward(self, x, node1):
        batch_size = x.size(0)
        mask = None
        q = MyModel1(node1)
        Q = q.q(x)
        K = q.k(x)
        V = q.v(x)
        Q = Q.view(batch_size, -1, q.num_heads, q.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, q.num_heads, q.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, q.num_heads, q.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (q.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention, V)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, q.d_model)
        weighted_values = q.WO(weighted_values)
        cnnz = self.attcnn(weighted_values)
        finalz = cnnz.squeeze(dim=1)
        return finalz
class MyModel1(nn.Module):
    def __init__(self, node1, num_heads=64):
        super(MyModel1, self).__init__()
        self.q = nn.Linear(node1, node1)
        self.k = nn.Linear(node1, node1)
        self.v = nn.Linear(node1, node1)
        self.WO = nn.Linear(node1, node1)
        self.d_model = node1
        self.num_heads = num_heads
        self.head_dim = node1 // num_heads
        assert self.head_dim * num_heads == node1, "d_model must be divisible by num_heads"
    def forward(self, x):
        return self.q(x)
class MyModel2(nn.Module):
    def __init__(self, node1):
        super(MyModel2, self).__init__()
        self.out = nn.Linear(node1, out_features=1)
    def forward(self, x):
        return self.out(x)
def compute_adjacency_matrix_from_cosine(cosNode, threshold = 0.31):
    adjacency_matrix = (cosNode > threshold).float()
    return adjacency_matrix