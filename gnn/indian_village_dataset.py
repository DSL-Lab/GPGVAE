import csv
from sklearn.preprocessing import OneHotEncoder
import sklearn
from torch_geometric.data import Data
from utils import to_dense_adj
import torch
import numpy as np

class IndianVillageGames(torch.utils.data.Dataset):
    def __init__(self, root):
        super(IndianVillageGames, self).__init__()
        
        self.root = root
        data = []
        n_villages = 77
        enc = OneHotEncoder()
        enc.fit([
            ['1', '1'],
            ['2', '2'],
            ['3', '3'],
        ])

        for i in range(1, n_villages + 1):
            if i == 48:
                # Village n. 48 has an issue in the data (i.e. a value of 5 appearing as a categorical feature in the fourth column where only 1, 2 and 3 should appear)
                continue
            try:
                with open(f"{self.root}/hhcovariates{i}.csv", newline='') as csvfile:
                    continuous_features_matrix = []
                    categorical_features_matrix = []
                    reader = csv.reader(csvfile, delimiter='\t')
                    for row in reader:
                        continuous_features = [float(row[0]), float(row[1]), float(row[4]), float(row[5])]
                        continuous_features_matrix.append(continuous_features)            

                        categorical_features = [row[2], row[3]]
                        categorical_features_matrix.append(categorical_features)

                    categorical_features_matrix = enc.transform(categorical_features_matrix).todense()
                    x = np.concatenate([continuous_features_matrix, categorical_features_matrix], axis=1)
                    # x = sklearn.preprocessing.normalize(x, axis=0)

                with open(f"{self.root}/adj_allVillageRelationships_HH_vilno_{i}.csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    adj = []
                    for row in reader:
                        adj_vector = [int(x) for x in row]
                        adj.append(adj_vector)

                adj_correlation = torch.Tensor(np.corrcoef(x)).float()
                adj_correlation = adj_correlation.fill_diagonal_(0.0)
                adj = torch.FloatTensor(adj).fill_diagonal_(1.0)
                edge_index = torch.nonzero(adj).t().contiguous()
                X = torch.FloatTensor(x)
                A = torch.squeeze(to_dense_adj(edge_index))
                data.append(
                    {
                        "X": X,
                        "A": torch.FloatTensor(A)
                    }
                )
            except FileNotFoundError as e:
                continue

        self.data_list = data

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)
