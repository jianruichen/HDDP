import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform
from script.hgcn.layers.hyplayers import HGCNConv, HypGRU, HGATConv
from script.hgcn.manifolds import PoincareBall
from script.models.BaseModel import BaseModel

class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size=64):
        super(Attention, self).__init__()
        self.project = nn.Sequential(nn.Linear(emb_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        z = torch.tensor(z)
        w = self.project(z) # n * 1 # [265 2 32]    [6311 2 16]
        beta = torch.softmax(w, dim=1)
        x = (beta * z)
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc = nn.Linear(n_class, n_hid)

    def forward(self, x):
        x = self.fc(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, edim):
        super(GATLayer, self).__init__()
        self.a=Parameter(torch.FloatTensor(16, 16), requires_grad=True)
        self.a.data.uniform_(-(1/edim), (1/edim))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, c0):
        a = F.tanh(torch.mm(c0,self.a).to(self.device))
        a_exp = torch.exp(a).to(self.device)
        return a_exp


class HTGN(BaseModel):
    def __init__(self, args):
        super(HTGN, self).__init__(args)
        self.manifold_name = args.manifold
        self.manifold = PoincareBall()
        self.weight = nn.Parameter(torch.Tensor(args.nhid, args.nhid), requires_grad=True)
        self.project = nn.Sequential(nn.Linear(args.nhid, args.nhid), nn.ReLU(), nn.Linear(args.nhid, args.nhid, bias=False))
        self.c = Parameter(torch.ones(3, 1) * args.curvature, requires_grad=not args.fixed_curvature)
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nout)
        self.linear1 = nn.Linear(args.nfeat, args.nfeat)
        self.linear2 = nn.Linear(args.nfeat, args.nout)
        self.hidden_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        self.fc = HGNN_classifier(args.nhid, args.nhid)
        self.att = GATLayer(args.nfeat)
        self.use_hta = args.use_hta
        if args.aggregation == 'deg':
            self.layer1 = HGCNConv(self.manifold, 2 * args.nout, 2 * args.nhid, self.c[0], self.c[1],
                                   dropout=args.dropout)
            self.layer2 = HGCNConv(self.manifold, 2 * args.nhid, args.nout, self.c[1], self.c[2], dropout=args.dropout)
        if args.aggregation == 'att':
            self.layer1 = HGATConv(self.manifold, 2 * args.nout, 2 * args.nhid, self.c[0], self.c[1],
                                   heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=True)
            self.layer2 = HGATConv(self.manifold, 2 * args.nhid * args.heads, args.nout, self.c[1], self.c[2],
                                   heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=False)
        self.gru = nn.GRUCell(args.nout, args.nout)

        self.nhid = args.nhid
        self.nout = args.nout
        self.cat = True
        self.Q = Parameter(torch.ones((args.nout, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.num_window = args.nb_window
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.initHyperX(self.hidden_initial)] * self.num_window
        return self.hiddens

    def weighted_hiddens(self, hidden_window):
        if self.use_hta == 0:
            return self.manifold.proj_tan0(self.manifold.logmap0(self.hiddens[-1], c=self.c[2]), c=self.c[2])
        # temporal self-attention
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window, self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nout])
        s = torch.mean(a * hidden_window_new, dim=0) # torch.sum is also applicable
        return s

    def initHyperX(self, x, c=1.0):
        if self.manifold_name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x

    def htc(self, x, c):
        x = self.manifold.proj(x, c)
        h = self.manifold.proj(self.hiddens[-1], c)
        return self.manifold.sqdist(x, h, c).mean()

    def forward(self, edge_index, x=None, weight=None):
        if x is None:  # using trainable feat matrix
            x = self.initHyperX(self.linear(self.feat), self.c[0])
        else:
            c = F.leaky_relu(self.fc(self.weight))  #trainable curvature
            a = self.att(c)
            c = 1 / torch.abs(torch.sum((self.fc(a))))
            edge_index, adj = edge_index
            x_train = self.initHyperX(self.linear(x), c)
            output = self.DY(x_train).float()
            x_adj = self.linear(torch.tensor(adj).type(torch.float))
            x = self.initHyperX(x_adj.float(), c)
            x = self.manifold.mobius_add(output, x, c)
        if self.cat:
            x = torch.cat([x, self.hiddens[-1]], dim=1)

        # layer 1
        x = self.manifold.proj(x, c)
        x = self.layer1(x, edge_index, c)

        # layer 2
        x = self.manifold.proj(x, c)
        x = self.layer2(x, edge_index, c)

        # GRU layer
        x = self.toTangentX(x, c)  # to tangent space
        hlist = self.manifold.proj_tan0(
            torch.cat([self.manifold.logmap0(hidden, c) for hidden in self.hiddens], dim=0), c)
        h = self.weighted_hiddens(hlist)
        x = self.gru(x, h)  # can also utilize HypGRU
        x = self.toHyperX(x, c)  # to hyper space
        return x, c

    #Hyperbolic dynamics
    def DY(self, x_hyp):
        alpha = 0.8
        y = np.zeros((x_hyp.shape[0], x_hyp.shape[1]))
        y_a = np.zeros((x_hyp.shape[0], x_hyp.shape[1]))
        y_b = np.zeros((x_hyp.shape[0], x_hyp.shape[1]))
        y[0, :] = np.random.rand(x_hyp.shape[1]) * math.pi / 2
        wz = x_hyp.detach().numpy()
        for e in range(0, y.shape[0]-1):
            for i in range(0, y.shape[1]):
                a = 0
                b = 0
                bigger_3 = np.where(wz[i, :] > 0)
                a = a + sum(
                    math.tanh(x) for x in np.multiply(wz[i, bigger_3], (y[e, bigger_3]- y[e, i])).tolist()[0])

                smaller_3 = np.where(wz[i, :] < 0)
                b = b + sum(
                    math.tanh(x) for x in np.multiply(wz[i, smaller_3], (y[e, smaller_3] - y[e, i])).tolist()[0])

                y[e+1, i] = alpha * y[e, i] - (1 - alpha) * y[e, i] * y[e, i]
                y_a[e+1, i] = a
                y_b[e+1, i] = b

        K_1 = Attention.forward(self, y_a.astype(np.float32))
        K_2 = Attention.forward(self, y_b.astype(np.float32))
        output = K_1 + K_2
        return output