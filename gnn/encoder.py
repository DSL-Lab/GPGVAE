import torch
from torch import nn
from torch.nn import functional as F



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
