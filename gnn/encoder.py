import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.nn import GCNConv

def normalize_last_dimension(data):
    channel_means = np.mean(data, axis=-1, keepdims=True)
    channel_stds = np.std(data, axis=-1, keepdims=True)
    
    normalized_data = (data - channel_means) / (channel_stds + 1e-6)
    
    return normalized_data


class GCN_ZEncoder(nn.Module):
    def __init__(self, n_games, hidden_dim, n_nodes, n_layer = 2):
        super(GCN_ZEncoder, self).__init__()
        self.n_games = n_games
        self.hidden_dim = hidden_dim
        self.adj_A = nn.Parameter(torch.zeros(n_nodes, n_nodes), requires_grad=True)
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

        self.lin1 = torch.nn.Linear(self.n_games, self.hidden_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # X: [n_graphs, n_nodes, n_games]
    def forward(self, X):
        adj = self.adj_A
        node_feat = self.lin1(X[0])
        edge_index = torch.stack(torch.where(adj > 0), dim=-1).t()
        edge_weight = adj[edge_index[0, :], edge_index[1, :]]
        for conv, mlp in zip(self.convs, self.mlps):
            node_feat = conv(node_feat, edge_index, edge_weight)   # [num_node, d]
            node_feat = mlp(node_feat)
        z_mu = node_feat.unsqueeze(0)
        
        return z_mu, self.adj_A


class MLPEncoder(nn.Module):
    def __init__(self, n_games, hidden_dim):
        super(MLPEncoder, self).__init__()
        self.n_games = n_games
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        self.lin1 = torch.nn.Linear(self.n_games, self.hidden_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # X: [n_graphs, n_nodes, n_games]
    def forward(self, X, adj = None):
        h = F.relu(self.lin1(X)) # B X N X H
        Z_mu = self.mlp(h)  # B X N X H
        # Z_logvar = self.mlp2(h)  # B X N X H
        
        return Z_mu


class TransformerZEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_games):
        super(TransformerZEncoder, self).__init__()
        self.hidden_dim = hidden_dim   # H

        self.input = nn.Sequential(
            nn.Linear(n_games, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=0, batch_first=True), 
            num_layers=2
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )


    def forward(self, X, adj = None):
        X_ = self.input(X)
        X_ = self.encoder(X_)  # n_graphs, n_nodes, hidden_dim
        z_mu = self.mlp(X_)  # n_graphs, n_nodes, hidden_dim

        return z_mu
    

class MLPEncoder2(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_games, num_mix_component):
        super(MLPEncoder2, self).__init__()

        # self.n_nodes = n_nodes   # N
        self.num_mix_component = num_mix_component   # K
        self.hidden_dim = hidden_dim   # H
        self.n_group = 32

        self.input_g = nn.Linear(n_games, self.n_group * self.hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.n_group, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.logit_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_mix_component))

        self.logit_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_mix_component))


    def forward(self, X):
        n_graphs, self.n_nodes, n_games = X.shape[0], X.shape[1], X.shape[2] 
        X = self.input_g(X).view(n_graphs, self.n_nodes, self.n_group, self.hidden_dim)
        X = X.permute(0, 2, 1, 3).view(-1, self.n_nodes, self.hidden_dim)  # n_graphs*n_group, n_nodes, hidden_dim
        att = torch.bmm(X, X.transpose(1, 2)) / self.hidden_dim
        att = att.view(n_graphs, self.n_group, self.n_nodes * self.n_nodes).permute(0, 2, 1)

        z_tilde = self.mlp(att).view(n_graphs, self.n_nodes, self.n_nodes, self.hidden_dim)        
        logit_theta = self.logit_theta(z_tilde)   # n_graphs, n_nodes, n_nodes, num_mix_component
        logit_alpha = self.logit_alpha(z_tilde)  # n_graphs, n_nodes, n_nodes, num_mix_component
        logit_alpha = logit_alpha.view(n_graphs, -1, self.num_mix_component) # n_graphs, n_nodes*n_nodes, num_mix_component
        logit_alpha = logit_alpha.mean(dim = 1) # n_graphs, num_mix_component
        logit_theta = (logit_theta + logit_theta.transpose(1, 2))/2

        return logit_theta, logit_alpha


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_games, num_mix_component):
        super(TransformerEncoder, self).__init__()

        self.num_mix_component = num_mix_component  # K
        self.hidden_dim = hidden_dim   # H

        self.input = nn.Sequential(
            nn.Linear(n_games, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=0, batch_first=True), 
            num_layers=2
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.logit_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_mix_component))
            
        self.logit_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component))
        
    def forward(self, X):
        n_graphs, self.n_nodes, n_games = X.shape[0], X.shape[1], X.shape[2]
        X_ = self.input(X)
        X_ = self.encoder(X_)  # n_graphs, n_nodes, hidden_dim

        Z_from = X_.unsqueeze(1).repeat(1, self.n_nodes, 1, 1)  # n_graphs, n_nodes, n_nodes, n_feat
        Z_to = X_.unsqueeze(2).repeat(1, 1, self.n_nodes, 1)
        z_tilde = (self.mlp((Z_from-Z_to)) + self.mlp((Z_to-Z_from)))/2 # n_graphs, n_nodes, n_nodes, hidden_dim
        
        logit_theta = self.logit_theta(z_tilde)   # n_graphs, n_nodes, n_nodes, num_mix_component  
        logit_theta = (logit_theta + logit_theta.transpose(1, 2))/2 
        logit_alpha = self.logit_alpha(z_tilde)  # n_graphs, n_nodes, n_nodes, num_mix_component
        logit_alpha = logit_alpha.view(n_graphs, -1, self.num_mix_component) # n_graphs, n_nodes*n_nodes, num_mix_component
        logit_alpha = logit_alpha.mean(dim = 1) # n_graphs, num_mix_component

        return logit_theta, logit_alpha


class PerGameTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_mix_component):
        super(PerGameTransformerEncoder, self).__init__()

        # self.n_nodes = n_nodes   # N
        self.num_mix_component = 1   # K
        self.hidden_dim = hidden_dim   # H

        # NOTE Muchen: n_head cannot set too large 10 -> 2
        self.n_heads = n_heads    

        self.lin = nn.Linear(1, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.W_Q = nn.Linear(self.hidden_dim, self.hidden_dim*self.n_heads, bias=False)
        self.W_K = nn.Linear(self.hidden_dim, self.hidden_dim*self.n_heads, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_heads + self.hidden_dim, self.hidden_dim * self.n_heads),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * self.n_heads, self.hidden_dim)
        )
        self.linF = nn.Linear(self.hidden_dim, self.hidden_dim)  

        self.logit_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component))
        self.logit_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component))


    def forward(self, X):
        n_graphs, self.n_nodes, n_games = X.shape[0], X.shape[1], X.shape[2] 
        X_ = X   # n_graphs, n_nodes, n_games
        X_ = X_.unsqueeze(3)   # n_graphs, n_nodes, n_games, 1
        X_ = self.lin(X_)  # n_graphs, n_nodes, n_games, hidden_dim
        
        
        m1 = self.W_Q(X_).reshape(n_graphs, self.n_nodes, n_games, self.n_heads, self.hidden_dim).permute(0, 3, 1, 2, 4)  # n_graphs, n_heads, n_nodes, n_games, hidden_dim
        m1 = m1.reshape(n_graphs*self.n_heads, self.n_nodes, -1)  # n_graphs*n_heads, n_nodes, n_games*hidden_dim
        m2 = self.W_K(X_).reshape(n_graphs, self.n_nodes, n_games, self.n_heads, self.hidden_dim).permute(0, 3, 2, 4, 1)   # n_graphs, n_heads, n_games, hidden_dim, n_nodes
        m2 = m2.reshape(n_graphs*self.n_heads, -1, self.n_nodes)   # n_graphs*n_heads, n_games*hidden_dim, n_nodes

        a_tilde = torch.bmm(m1, m2).reshape(n_graphs, self.n_heads, self.n_nodes, self.n_nodes)  # n_graphs, n_heads, n_nodes, n_nodes
        a = torch.softmax(a_tilde/(np.sqrt(n_games*self.hidden_dim)), dim=3)  # n_graphs, n_heads, n_nodes, n_nodes
        a = a.reshape(n_graphs, self.n_heads*self.n_nodes, self.n_nodes) # n_graphs, n_heads*n_nodes, n_nodes

        X__ = X_.view(n_graphs, self.n_nodes, -1)  # n_graphs, n_nodes, n_games*hidden_dim
        s = torch.bmm(a, X__).reshape(n_graphs, self.n_heads, self.n_nodes, n_games, -1)  # n_graphs,n_heads, n_nodes, n_games, hidden_dim
        s = s.permute(0, 2, 3, 1, 4).reshape(n_graphs, self.n_nodes, n_games, -1)  # n_graphs, n_nodes, n_games, n_heads*hidden_dim

        Z = self.linF(X_) + self.mlp(torch.cat([X_, s], dim=3))  # n_graphs, n_nodes, n_games, n_heads*hidden_dim+hidden_dim  => n_graphs, n_nodes, n_games, hidden_dim
        Z = self.layer_norm(Z)  # n_graphs, n_nodes, n_games, hidden_dim
        n_feat = Z.shape[-1]   # n_graphs, n_nodes, n_games, hidden_dim

        
        Z = Z.permute(0, 3, 1, 2).reshape(n_graphs*n_feat, self.n_nodes, n_games)  # n_graphs*n_feat, n_nodes, n_games
        z_tilde = torch.bmm(Z, Z.transpose(1, 2))/n_games  # n_graphs*n_feat, n_nodes, n_nodes
        z_tilde = z_tilde.reshape(n_graphs, n_feat, self.n_nodes, self.n_nodes).permute(0, 2, 3, 1)
        logit_theta = self.logit_theta(z_tilde)   # n_graphs, n_nodes, n_nodes, num_mix_component
        logit_alpha = self.logit_alpha(z_tilde)  # n_graphs, n_nodes, n_nodes, num_mix_component
        logit_alpha = logit_alpha.view(n_graphs, -1, self.num_mix_component) # n_graphs, n_nodes*n_nodes, num_mix_component
        logit_alpha = logit_alpha.mean(dim = 1) # n_graphs, num_mix_component
        logit_theta = logit_theta.squeeze(-1)
        return logit_theta, logit_alpha
    


# class PerGameTransformerEncoder(nn.Module):
#     def __init__(self, hidden_dim, n_heads,num_mix_component):
#         super(PerGameTransformerEncoder, self).__init__()

#         # self.n_nodes = n_nodes   # N
#         self.num_mix_component = num_mix_component   # K
#         self.hidden_dim = hidden_dim   # H

#         # NOTE Muchen: n_head cannot set too large 10 -> 2
#         self.n_heads = n_heads    

#         self.lin = nn.Linear(2000, self.hidden_dim)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#         self.input = nn.Sequential(
#             nn.Linear(1, self.hidden_dim//4),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_dim//4, self.hidden_dim)
#         )
#         self.input_att = nn.Sequential(
#             nn.Linear(1, self.hidden_dim//4),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_dim//4, self.hidden_dim)
#         )

#         self.W_Q = nn.Linear(self.hidden_dim, self.hidden_dim*self.n_heads, bias=False)
#         self.W_K = nn.Linear(self.hidden_dim, self.hidden_dim*self.n_heads, bias=False)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.hidden_dim * self.n_heads + self.hidden_dim, self.hidden_dim * self.n_heads),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_dim * self.n_heads, self.hidden_dim)
#         )
#         self.linF = nn.Linear(self.hidden_dim, self.hidden_dim)  

#         self.logit_theta = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_dim, self.num_mix_component))
#         self.logit_alpha = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_dim, self.num_mix_component))


#     def forward(self, X):
#         n_graphs, self.n_nodes, n_games = X.shape[0], X.shape[1], X.shape[2] 
#         # X_ = X   # n_graphs, n_nodes, n_games
#         # X = X.unsqueeze(-1)   # n_graphs, n_nodes, n_games, 1
#         # att = self.input_att(X)  # n_graphs, n_nodes, n_games, hidden_dim
#         # att = att.sum(dim=-2, keepdim=False)/n_games
#         X_ = F.relu(self.lin(X))  # n_graphs, n_nodes, hidden_dim
        
        
#         m1 = self.W_Q(X_).reshape(n_graphs, self.n_nodes, self.n_heads, self.hidden_dim).permute(0, 2, 1, 3)  # n_graphs, n_heads, n_nodes, hidden_dim
#         m1 = m1.reshape(n_graphs*self.n_heads, self.n_nodes, -1)  # n_graphs*n_heads, n_nodes, n_games*hidden_dim
#         m2 = self.W_K(X_).reshape(n_graphs, self.n_nodes, self.n_heads, self.hidden_dim).permute(0, 2, 3, 1)   # n_graphs, n_heads, hidden_dim, n_nodes
#         m2 = m2.reshape(n_graphs*self.n_heads, -1, self.n_nodes)   # n_graphs*n_heads, n_games*hidden_dim, n_nodes

#         a_tilde = torch.bmm(m1, m2).reshape(n_graphs, self.n_heads, self.n_nodes, self.n_nodes)  # n_graphs, n_heads, n_nodes, n_nodes
#         a = torch.softmax(a_tilde/(np.sqrt(self.hidden_dim)), dim=3)  # n_graphs, n_heads, n_nodes, n_nodes
#         a = a.reshape(n_graphs, self.n_heads*self.n_nodes, self.n_nodes) # n_graphs, n_heads*n_nodes, n_nodes

#         X__ = X_.view(n_graphs, self.n_nodes, -1)  # n_graphs, n_nodes, hidden_dim
#         s = torch.bmm(a, X__).reshape(n_graphs, self.n_heads, self.n_nodes, -1)  # n_graphs,n_heads, n_nodes, hidden_dim
#         s = s.permute(0, 2, 1, 3).reshape(n_graphs, self.n_nodes, -1)  # n_graphs, n_nodes, n_heads*hidden_dim

#         Z = self.linF(X_) + self.mlp(torch.cat([X_, s], dim=2))  # n_graphs, n_nodes, n_games, n_heads*hidden_dim+hidden_dim  => n_graphs, n_nodes, n_games, hidden_dim
#         Z = self.layer_norm(Z)  # n_graphs, n_nodes, hidden_dim
#         n_feat = Z.shape[-1]   # n_graphs, n_nodes, hidden_dim

        
#         Z = Z.unsqueeze(3).permute(0, 2, 1, 3).reshape(n_graphs*n_feat, self.n_nodes, -1)  # n_graphs*n_feat, n_nodes
#         z_tilde = torch.bmm(Z, Z.transpose(1, 2))  # n_graphs*n_feat, n_nodes, n_nodes
#         z_tilde = z_tilde.reshape(n_graphs, n_feat, self.n_nodes, self.n_nodes).permute(0, 2, 3, 1)
#         logit_theta = self.logit_theta(z_tilde)   # n_graphs, n_nodes, n_nodes, num_mix_component
#         logit_alpha = self.logit_alpha(z_tilde)  # n_graphs, n_nodes, n_nodes, num_mix_component
#         logit_alpha = logit_alpha.view(n_graphs, -1, self.num_mix_component) # n_graphs, n_nodes*n_nodes, num_mix_component
#         logit_alpha = logit_alpha.mean(dim = 1) # n_graphs, num_mix_component
        
#         return logit_theta, logit_alpha
