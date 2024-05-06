import time
import pickle
import math
import numpy as np
import networkx as nx
from numpy.linalg import inv, pinv
import random
import sklearn
import sklearn.preprocessing
import torch
from torch_geometric.data import Data
from small_model import to_dgl


## generate synthetic datasets and then use them to train the interaction encoder at the first stage
class Pretrain_Games(torch.utils.data.Dataset):
    def __init__(self, root, n_graphs, n_games, m=3, p=0.6, signal_to_noise_ratio=10, regenerate_data=True, seed = 1):
        super(Pretrain_Games, self).__init__()
    
        self.n_graphs = n_graphs
        self.m = m
        self.p = p
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.data_list = None
        self.seed = seed
        self.n_games = n_games

        file_name = f'/data_pretraining_{n_graphs}.pkl'
        if not regenerate_data:
            try:
                self.data_list = pickle.load(open(root + file_name, "rb"))
                print(f"Loaded pre-existing graphs")
            except:
                print("Graphs not found, generating them")
        
        if regenerate_data or self.data_list is None:
            start = time.time()
            print("Generating graphs")

            networkdata = []
            for j in range(self.n_graphs):
                self.alpha = np.random.uniform(0.0, 1)
                self.target_spectral_radius = np.random.uniform(-0.9, 0.9)
                self.game_type = random.choice(['linear_quadratic', 'linear_influence', 'barik_honorio'])
                self.graph_type = random.choice(['barabasi_albert', 'erdos_renyi', 'watts_strogatz'])
                self.n_nodes = np.random.randint(10, 50)
                data = self.generate_network_game()
                networkdata.append(data)

            self.data_list = networkdata
           
            pickle.dump(self.data_list, open(root + file_name, "wb"))
            print(f"Finished generating graphs. It took {time.time() - start}s")


    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

    def get_graph(self, graph_type):
        if graph_type == "barabasi_albert":
            return nx.barabasi_albert_graph(self.n_nodes, self.m)
        elif graph_type == "erdos_renyi":
            return nx.erdos_renyi_graph(self.n_nodes, self.p, directed=False)
        elif graph_type == "watts_strogatz":
            k = math.floor(np.log2(self.n_nodes))
            return nx.watts_strogatz_graph(self.n_nodes, k=k, p=self.p)

    def generate_network_game(self):
        # Adjacency matrix and laplacian are symmetric
        G = self.get_graph(self.graph_type)
        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G).todense()   # L = D - A, L = D^(-1/2) L D^(-1/2) = I - D^(-1/2) W D^(-1/2)
        A = nx.adjacency_matrix(G).todense()
        A_norm = np.eye(self.n_nodes) - L  # A_norm = D^(-1/2) W D^(-1/2)  # N X N
        L_filtered =np.eye(self.n_nodes) - self.alpha * A_norm
       
        B_covariance = pinv(L_filtered) 
        B = np.random.multivariate_normal(mean = np.zeros(self.n_nodes), cov = B_covariance, size = self.n_games).T
   
        noise_variance = 1 / (self.signal_to_noise_ratio * self.n_nodes)
        noise = np.random.multivariate_normal(mean = np.zeros(self.n_nodes), cov = noise_variance * np.eye(self.n_nodes), size = self.n_games).T
        X = self.get_actions(A_norm=A_norm, L=L, B=B) + noise

        edges = np.array(G.edges())
        X = torch.from_numpy(X).float()
        B = torch.from_numpy(B).float()
        A = torch.FloatTensor(A)    
        G = Data(x = X, edge_index = torch.tensor(edges).T) 
        G = to_dgl(G)
                
        return {"X": X, "A": A, "g": G}

    def get_actions(self, A_norm, L, B):
        spectral_radius = max([abs(i) for i in np.linalg.eigvals(A_norm)])
        beta = self.target_spectral_radius
        spectral_radius_of_betaG = beta * spectral_radius

        assert abs(spectral_radius_of_betaG) <= 1, f"spectral_radius_of_betaG: {spectral_radius_of_betaG} is outside the [-1, 1] range"

        if self.game_type == "linear_influence":
            actions = np.matmul(pinv(A_norm), B)
        elif self.game_type == "barik_honorio":
            _, eigenvectors = np.linalg.eigh(A_norm)
            # Actions are the eigenvector with eigenvalue equal to 1, i.e. the last eigenvector
            last_eigenvector = np.asarray(eigenvectors[:, -1]).squeeze()
            actions = []
            eps = 0.2
            count = 0
            while len(actions) < self.n_games:
                count += 1
                action = last_eigenvector + np.random.normal(loc=0, scale=1, size=(last_eigenvector.shape[0]))
                if np.linalg.norm(action - A_norm @ action) / self.n_nodes <= eps:
                    actions.append(action)
            actions = np.array(actions).T
        elif self.game_type == "linear_quadratic":
            actions = np.matmul(inv(np.identity(self.n_nodes) - beta * A_norm), B)
        else:
            raise NotImplementedError(f"Game type {self.game_type} not implemented")

        normalized_actions = sklearn.preprocessing.normalize(np.asarray(actions), axis=0)  ## features
        return normalized_actions         


class Games(torch.utils.data.Dataset):
    def __init__(self, root, n_graphs, n_games=200, n_nodes=50, m=3, p=0.6, target_spectral_radius=0.9, alpha=1.0, signal_to_noise_ratio=10, graph_type="barabasi_albert", game_type="linear_quadratic", regenerate_data=True, seed = 1):
        super(Games, self).__init__()
    
        self.n_graphs = n_graphs
        self.n_games = n_games
        self.n_nodes = n_nodes
        self.m = m
        self.p = p
        self.target_spectral_radius = target_spectral_radius
        self.alpha = alpha
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.graph_type = graph_type
        self.game_type = game_type
        self.data_list = None
        self.seed = seed

        file_name = f"/data_{graph_type}_{game_type}_{target_spectral_radius}_{alpha}_{n_graphs}_{n_games}_{n_nodes}.pkl"
        if not regenerate_data:
            try:
                self.data_list = pickle.load(open(root + file_name, "rb"))
                print(f"Loaded pre-existing graphs")
            except:
                print("Graphs not found, generating them")
        
        if regenerate_data or self.data_list is None:
            start = time.time()
            print("Generating graphs")

            self.data_list = []
            for j in range(self.n_graphs):
                data = self.generate_network_game()
                self.data_list.append(data)

            pickle.dump(self.data_list, open(root + file_name, "wb"))
            print(f"Finished generating graphs. It took {time.time() - start}s")


    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

    def get_graph(self, graph_type):
        if graph_type == "barabasi_albert":
            return nx.barabasi_albert_graph(self.n_nodes, self.m)
        elif graph_type == "erdos_renyi":
            return nx.erdos_renyi_graph(self.n_nodes, self.p, directed=False)
        elif graph_type == "watts_strogatz":
            k = math.floor(np.log2(self.n_nodes))
            return nx.watts_strogatz_graph(self.n_nodes, k=k, p=self.p)


    def generate_network_game(self):
        # Adjacency matrix and laplacian are symmetric
        G = self.get_graph(self.graph_type)

        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G).todense()   # L = D - A, L = D^(-1/2) L D^(-1/2) = I - D^(-1/2) W D^(-1/2)
        A = nx.adjacency_matrix(G).todense()
        A_norm = np.eye(self.n_nodes) - L  # A_norm = D^(-1/2) W D^(-1/2)  # N X N
        L_filtered =np.eye(self.n_nodes) - self.alpha * A_norm
       
        B_covariance = pinv(L_filtered) 
        B = np.random.multivariate_normal(mean = np.zeros(self.n_nodes), cov = B_covariance, size = self.n_games).T       

        noise_variance = 1 / (self.signal_to_noise_ratio * self.n_nodes)
        noise = np.random.multivariate_normal(mean = np.zeros(self.n_nodes), cov = noise_variance * np.eye(self.n_nodes), size = self.n_games).T
        X = self.get_actions(A_norm=A_norm, L=L, B=B) + noise

        edges = np.array(G.edges())
        X = torch.from_numpy(X).float()
        B = torch.from_numpy(B).float()
        A = torch.FloatTensor(A)    

        print(f'number of users: {X.shape[0]}, number of businesses: {X.shape[1]}')
        G = Data(x = X, edge_index = torch.tensor(edges).T) 
        G = to_dgl(G)    
        
        return {"X": X, "A": A, "g": G}

        
    def get_actions(self, A_norm, L, B):
        spectral_radius = max([abs(i) for i in np.linalg.eigvals(A_norm)])
        beta = self.target_spectral_radius
        spectral_radius_of_betaG = beta * spectral_radius
        assert abs(spectral_radius_of_betaG) <= 1, f"spectral_radius_of_betaG: {spectral_radius_of_betaG} is outside the [-1, 1] range"

        if self.game_type == "linear_influence":
            actions = np.matmul(pinv(A_norm), B)
        elif self.game_type == "barik_honorio":
            _, eigenvectors = np.linalg.eigh(A_norm)
            # Actions are the eigenvector with eigenvalue equal to 1, i.e. the last eigenvector
            last_eigenvector = np.asarray(eigenvectors[:, -1]).squeeze()
            actions = []
            eps = 0.2
            count = 0
            while len(actions) < self.n_games:
                count += 1
                action = last_eigenvector + np.random.normal(loc=0, scale=1, size=(last_eigenvector.shape[0]))
                if np.linalg.norm(action - A_norm @ action) / self.n_nodes <= eps:
                    actions.append(action)
            actions = np.array(actions).T
            # print(f"It took {count} trials to generate {self.n_games} valid games")
        elif self.game_type == "linear_quadratic":
            actions = np.matmul(inv(np.identity(self.n_nodes) - beta * A_norm), B)
        else:
            raise NotImplementedError(f"Game type {self.game_type} not implemented")

        normalized_actions = sklearn.preprocessing.normalize(np.asarray(actions), axis=0)  ## features

        return normalized_actions

