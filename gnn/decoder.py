import torch 
import torch.nn.functional as F
import torch_geometric
# from torch_geometric.nn.conv import GCNConv
from torch import nn
import networkx as nx


class DotProductDecoder(torch.nn.Module):
    def forward(self, z):
        A = z @ torch.transpose(z, 1, 2)
        return A

class MLPDecoder(torch.nn.Module):
    def __init__(self, in_channels, permutation_invariant=False):
        super(MLPDecoder, self).__init__()
        if permutation_invariant:
            self.lin = torch.nn.Linear(in_channels, in_channels)
            self.lin_final = torch.nn.Linear(in_channels, 1)
        else:
            self.lin = torch.nn.Linear(in_channels, in_channels)
            self.lin_final = torch.nn.Linear(in_channels, 1)
        self.permutation_invariant = permutation_invariant

    def forward(self, z):

        if self.permutation_invariant:
            n_graphs, n_nodes, n_games, n_feat = z.shape
            z = z.permute(0, 3, 1, 2).reshape(n_graphs*n_feat, n_nodes, n_games)  # n_graphs*n_feat, n_nodes, n_games
            z_tilde = torch.bmm(z, z.transpose(1, 2))  # n_graphs*n_feat, n_nodes, n_nodes
            z_tilde = z_tilde.reshape(n_graphs, n_feat, n_nodes, n_nodes).permute(0, 2, 3, 1)

            h = self.lin(z_tilde).relu()
            A = self.lin_final(h).reshape(-1, n_nodes, n_nodes)
        else:
            n_nodes = z.shape[1]
            nodes = torch.arange(start=0, end=n_nodes, dtype=torch.long)
            all_src = torch.repeat_interleave(nodes, n_nodes)
            all_dst = nodes.repeat(n_nodes)

            z_src = z[:, all_src]
            z_dst = z[:, all_dst]

            h = self.lin(z_src * z_dst).relu()
            A = self.lin_final(h).reshape(-1, n_nodes, n_nodes)

        return A

class empiricalDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.):
        super(empiricalDecoder, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, out_channels)
        # self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(out_channels)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, X):
        X = X[0]
        emp_cov = torch.cov(X.T)
        emp_cov.fill_diagonal_(0.)
        emp_cov = emp_cov * X.size(0) / emp_cov.sum()  
        
        h = F.relu(self.lin1(X))
        mu = F.relu(self.lin2(h))
        return self.batch_norm(mu)

class CorrelationCoefficientDecoder(nn.Module):
    def forward(self, z):
        # Batched matrix multiplication
        z_mean = torch.mean(z, dim=-1, keepdim=True)
        z_shift = z - z_mean
        cov = z_shift @ z_shift.transpose(1, 2)
        z_var = (z_shift * z_shift).sum(dim=-1, keepdim=True)
        corr_coeff = cov / torch.sqrt(z_var * z_var.transpose(1, 2))
        return (corr_coeff + 1) / 2  # shifts it in the range [0, 1] so that we can use MSE for training the model


class CosineSimilarityDecoder(nn.Module):
    def forward(self, z, pearson=False):
        # Batched matrix multiplication
        A = z @ torch.transpose(z, 1, 2)
        z_norm = torch.norm(z, dim=-1, keepdim=True)
        A = A / (z_norm * torch.transpose(z_norm, 1, 2))
        return A


from torch_geometric.nn import GCNConv

class GCNDecoder(nn.Module):
    def __init__(self, hidden_dim, out_channels, n_layer):
        super(GCNDecoder, self).__init__()
        self.nlayer = n_layer
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layer)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )
            for _ in 
            range(n_layer)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )
            
        self.lin = nn.Linear(hidden_dim, out_channels)  
    
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)
    
    def forward(self, node_feat, adj):
        assert node_feat.shape[0] == 1
        node_feat = node_feat[0]
        adj = adj[0]
        edge_index = torch.stack(torch.where(adj > 0), dim=-1).t()
        edge_weight = adj[edge_index[0, :], edge_index[1, :]]
        for conv, mlp in zip(self.convs, self.mlps):
            node_feat = conv(node_feat, edge_index, edge_weight)   # [num_node, d]
            node_feat = mlp(node_feat)
        mu = self.lin(node_feat).unsqueeze(0)
        return mu





# class GCNConv(nn.Module):
#     '''
#     math::
#         \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
#         \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

#     where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
#     adjacency matrix with inserted self-loops and
#     :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
#     The adjacency matrix can include other values than :obj:`1` representing
#     edge weights via the optional :obj:`edge_weight` tensor.  
#     '''

#     def __init__(self, hidden_size):
#         super(GCNConv, self).__init__()
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size)
#         )
     
#     def forward(self, X, A):
#         # identity = (torch.eye(A.shape[1], dtype=bool, device=A.device).repeat(A.shape[0],1,1)).float()   # B, N, N
#         identity = torch.eye(A.shape[0], device=A.device).float()
#         hatA = identity + A       # B, N, N
#         diags = hatA.sum(axis=1)  # B, N
#         diags = 1.0 / torch.sqrt(diags + 1e-3)
#         D = torch.diag_embed(diags)    # B, N, N
#         xe = D @ (hatA @ D) @ X      # B, N, d
#         y = hatA @ self.ffn(xe)             # B, N, d
#         return y


from torch_geometric.nn import ResGatedGraphConv

class GatedGCNDecoder(nn.Module):
    def __init__(self, hidden_dim, out_channels, n_layer):
        super(GatedGCNDecoder, self).__init__()
        self.nlayer = n_layer
        # self.conv1 = GCNConv(hidden_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, out_channels)
        self.convs = nn.ModuleList(
            [
                ResGatedGraphConv(hidden_dim, hidden_dim)
            for _ in 
            range(n_layer)]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )
            for _ in 
            range(n_layer)]
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )
            
        self.lin = nn.Linear(hidden_dim, out_channels)  
        # self.lin2 = nn.Linear(hidden_dim, out_channels) 
        
    def forward(self, node_feat, adj):
        assert node_feat.shape[0] == 1
        node_feat = node_feat[0]
        adj = adj[0]
        edge_index = torch.stack(torch.where(adj > 0), dim=-1).t()   #  loss gradient?
        # import ipdb; ipdb.set_trace()
        for conv, mlp in zip(self.convs, self.mlps):
            node_feat = conv(node_feat, edge_index)   # [num_node, d]
            node_feat = mlp(node_feat)
        mu = self.lin(node_feat).unsqueeze(0)
        # xe = self.conv1(node_feat, adj)
        # xe = F.relu(xe)
        # y = self.conv2(xe, adj)
        logvar = self.mlp(node_feat).unsqueeze(0)

        return mu, logvar