import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling
from dgl import function as fn


class IntegerFeatureEncoder(torch.nn.Module):
    r"""Provides an encoder for integer node features.

    Args:
        emb_dim (int): The output embedding dimension.
        num_classes (int): The number of classes/integers.

    Example:

        >>> encoder = IntegerFeatureEncoder(emb_dim=16, num_classes=10)
        >>> batch = torch.randint(0, 10, (10, 2))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_classes, emb_dim)
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, x):
        # Encode just the first dimension if more exist
        x = self.encoder(x[:, 0].to(torch.long))

        return x


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [B, N]
        # output: [B, N, d]
        # e^{i * -log10000/d} = 1/ (10000^{i/d})

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)  # [e, sin, cos] [B, N, d+1]

        return self.eig_w(eeig)  # [B, N, d]


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

    
class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Conv, self).__init__()

        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.preffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, graph, x_feat, bases):
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.apply_edges(fn.copy_u('x', '_x'))  # Builtin message function that computes message using source node feature.
            xee = self.pre_ffn(graph.edata['_x']) * bases  # [num_edge, d]
            graph.edata['v'] = xee                       # [num_edge, d]
            graph.update_all(fn.copy_e('v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
            y = graph.ndata['aggr_e']  # [num_node, d]
            y = self.preffn_dropout(y) # [num_node, d]
            x = x_feat + y             # [num_node, d]
            y = self.ffn(x)            # [num_node, d]
            y = self.ffn_dropout(y)    # [num_node, d]
            x = x + y                  # [num_node, d]
            return x


class SpecformerSmall(nn.Module):

    def __init__(self, nclass, nlayer, hidden_dim=128, nheads=4, feat_dropout=0.1, trans_dropout=0.1, adj_dropout=0.1):
        super(SpecformerSmall, self).__init__()
        
        print('small model')
        self.nlayer = nlayer
        self.nclass = nclass
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        # self.nfeatures = nfeatures

        # self.lin = nn.Linear(nfeatures, hidden_dim)

        # self.x_encoder = IntegerFeatureEncoder(emb_dim=hidden_dim, num_classes=3000)
        # self.x_encoder = nn.Embedding(nclass, hidden_dim)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(trans_dropout)
        self.ffn_dropout = nn.Dropout(trans_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, trans_dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.adj_dropout = nn.Dropout(adj_dropout)
        self.filter_encoder = nn.Sequential(
            nn.Linear(nheads + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
 
        self.convs = nn.ModuleList([Conv(hidden_dim, feat_dropout) for _ in range(nlayer)])
        # self.convs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(nlayer)])
        self.pool = AvgPooling()
        self.linear = nn.Linear(hidden_dim, nclass)
        

    def forward(self, e, u, g, length, edge_indices):

        # e: [B, N]        eigenvalues
        # u: [B, N, N]     eigenvectors
        # x: [B, N, d]     node features
        # f: [B, N, N, d]  edge features   

        ut = u.transpose(1, 2)  # [B, N, N]

        # do not use u to generate edge_idx because of the connected components
        e_mask, edge_idx = self.length_to_mask(edge_indices, length) # [B, N], [B, N, N]
        eig = self.eig_encoder(e)  # [B, N, d]

        # node_feat = self.x_encoder(node_feat.to(torch.long))     # [B*N, d]
        # node_feat = self.lin(node_feat)     # [B*N, d]
        node_feat = g.ndata['x'].to(e.device)  # [B*N, d]
        node_feat = torch.randn(node_feat.shape[0], eig.shape[-1]).to(e.device)   # random/ eigen vector/ constant eigen vector is size [B, N, N], still has dimension problem
        # node_feat = g.ndata['x']  # [B*N, d]
        # node_feat = torch.randn(node_feat.shape[0], eig.shape[-1])
        # node_feat = torch.ones(node_feat.shape[0], eig.shape[-1])   # random/ eigen vector/ constant

        mha_eig = self.mha_norm(eig)  # LN(Z) [B, N, d]
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=e_mask)  # MHA(LN(Z)) [B, N, d], [B, N, N]
        eig = eig + self.mha_dropout(mha_eig)   # Z = Z + MHA(LN(Z)) [B, N, d]

        ffn_eig = self.ffn_norm(eig)  # LN(Z)        [B, N, d]
        ffn_eig = self.ffn(ffn_eig)   # FFN(LN(Z))   [B, N, d]
        eig = eig + self.ffn_dropout(ffn_eig)  # Z = Z + FFN(LN(Z))    [B, N, d]

        new_e = self.decoder(eig).transpose(2, 1)      # [B, m, N]  m heads, N new eigenvalues 
        diag_e = torch.diag_embed(new_e)               # [B, m, N, N]

        identity = torch.diag_embed(torch.ones_like(e))
        bases = [identity]
        for i in range(self.nheads):
            filters = u @ diag_e[:, i, :, :] @ ut
            bases.append(filters)

        bases = torch.stack(bases, axis=-1)  # [B, N, N, H], H heads = m + 1     
        bases = bases[edge_idx]   # [num_edge, H]
        bases = self.adj_dropout(self.filter_encoder(bases))   # (5) [B, N, N, d]
        # bases = edge_softmax(g, bases)  
       
        for conv in self.convs:
            node_feat = conv(g, node_feat, bases)   # [num_node, d]


        h = self.pool(g, node_feat)    # [B, d]
        h = self.linear(h)           # [B, nclass]

        return h


    def length_to_mask(self, indices, length):
        '''
        length: [B]
        return: [B, max_len].
        '''
        B = len(indices)  # batch size
        N = length.max().item()  # max number of nodes in a batch
        mask1d = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)
        mask2d = torch.zeros(B, N, N, device=length.device)
        for i in range(B):
            mask2d[i, indices[i][0].long(), indices[i][1].long()] = 1.0

        mask2d = mask2d.bool()
        return mask1d, mask2d
    

def to_dgl(data):
    """Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
    object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The data object.

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
        >>> x = torch.randn(5, 3)
        >>> edge_attr = torch.randn(6, 2)
        >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes=5, num_edges=6,
            ndata_schemes={'x': Scheme(shape=(3,))}
            edata_schemes={'edge_attr': Scheme(shape=(2, ))})

        >>> data = HeteroData()
        >>> data['paper'].x = torch.randn(5, 3)
        >>> data['author'].x = torch.ones(5, 3)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        >>> data['author', 'cites', 'paper'].edge_index = edge_index
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes={'author': 5, 'paper': 5},
            num_edges={('author', 'cites', 'paper'): 5},
            metagraph=[('author', 'paper', 'cites')])
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, store in data.edge_items():
            if store.get('edge_index') is not None:
                row, col = store.edge_index
            else:
                row, col, _ = store['adj_t'].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, store in data.node_items():
            for attr, value in store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


class MLPClassification(nn.Module):
    def __init__(self, n_eigen, hidden_dim):
        super(MLPClassification, self).__init__()
        self.n_eigen = n_eigen
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2))

        self.lin1 = torch.nn.Linear(self.n_eigen, self.hidden_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, X):
        # sort_X = torch.stack([torch.sort(X[i], dim=0, descending=True)[0][:self.n_eigen] for i in range(X.shape[0])], dim=0)
        h1 = F.relu(self.lin1(X)) # N X H
        z = self.mlp(h1)  # N X H    
        return z
    