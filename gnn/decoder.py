import torch 
from torch import nn
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
        # assert node_feat.shape[0] == 1
        # node_feat = node_feat[0]
        # adj = adj[0]
        # edge_index = torch.stack(torch.where(adj > 0), dim=-1).t()
        # edge_weight = adj[edge_index[0, :], edge_index[1, :]]
        # for conv, mlp in zip(self.convs, self.mlps):
        #     node_feat = conv(node_feat, edge_index, edge_weight)   # [num_node, d]
        #     node_feat = mlp(node_feat)
        # mu = self.lin(node_feat).unsqueeze(0)
        n_graphs, n_nodes = node_feat.shape[0], node_feat.shape[1]
        new_adj = torch.zeros((n_graphs*n_nodes, n_graphs*n_nodes)).to(node_feat.device)
        for i in range(n_graphs):
            new_adj[i*n_nodes:(i+1)*n_nodes, i*n_nodes:(i+1)*n_nodes] = adj[i]
        edge_index = torch.stack(torch.where(new_adj > 0), dim=-1).t()
        edge_weight = new_adj[edge_index[0, :], edge_index[1, :]]

        node_feat = node_feat.reshape(n_graphs*n_nodes, node_feat.shape[2])

        for conv, mlp in zip(self.convs, self.mlps):
            node_feat = conv(node_feat, edge_index, edge_weight)   # [num_node, d]
            node_feat = mlp(node_feat)
        mu = self.lin(node_feat)
        mu = mu.reshape(n_graphs, n_nodes, -1)

        return mu