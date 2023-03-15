import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    """A gcn network with two layers of convolution blocks"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        :param nfeat: number of features
        :param nhid:  number of fearures for hidden layers
        :param nclass: number of classes for classfication
        :param dropout: the dropout probability
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        :param x: input matrix with the shape node_num*features
        :param adj: sparse adjacent matrix for edges
        return: class of each node
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
