import os.path as osp
import os
import torch
from games_dataset import Games
from indian_village_dataset import IndianVillageGames
import pickle
import dgl
from torch_geometric.data import Data
from small_model import to_dgl
from torch.utils.data import DataLoader


def data_preprocess(args):
    if args.game_type == 'village':
        data_name = 'indian_networks'      
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        dataset = IndianVillageGames(path)
        new_dataset = []
        for i in range(len(dataset)):
            X = dataset[i]['X']
            A = torch.FloatTensor(dataset[i]['A'])
            edges = A.nonzero().t()
            X = torch.FloatTensor(X)
            G = Data(x = X, edge_index = torch.tensor(edges))
            G = to_dgl(G)
            new_dataset.append({"X": X, "A": A, "g": G})      
            print(f'number of users: {X.shape[0]}, number of businesses: {X.shape[1]}')         

    elif args.game_type == 'Yelp':      
        data_name = os.path.join('Yelp', str(args.graph_type)+'_food.pickle')
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        with open(path, 'rb+') as f:
            dataset = pickle.load(f)
        new_dataset = []   
        
        for i in range(len(dataset)):
            X = []
            for xi in range(dataset[i]['X'].size(-1)):
                if len(torch.nonzero(dataset[i]['X'][:,xi])) > 0:
                    X.append(dataset[i]['X'][:,xi])

            X = torch.stack(X, dim = 1)
            X = torch.log(X + 0.001)
            X = torch.FloatTensor(X)
            A = torch.FloatTensor(dataset[i]['A'])         
            edges = A.nonzero().t()
            G = Data(x = X, edge_index = torch.tensor(edges)) 
            print(f'number of users: {X.shape[0]}, number of businesses: {X.shape[1]}')
            G = to_dgl(G)
            new_dataset.append({"X": X, "A": A, "g": G})      
        print(f'number of graphs: {len(new_dataset)}')
      
    elif args.game_type == 'Foursquare':
        data_name = os.path.join('Foursquare', str(args.graph_type)+'_dataset.pickle')
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        with open(path, 'rb+') as f:
            dataset = pickle.load(f)   
        new_dataset = []   
        for i in range(len(dataset)):
            X = torch.log(dataset[i]['X'] + 1e-3)
            A = torch.FloatTensor(dataset[i]['A'])  
            X = torch.FloatTensor(X)       
            edges = A.nonzero().t()
            G = Data(x = X, edge_index = torch.tensor(edges)) 
            print(f'number of users: {X.shape[0]}, number of businesses: {X.shape[1]}')
            G = to_dgl(G)
            new_dataset.append({"X": X, "A": A, "g": G})      

    elif args.graph_type == 'barabasi_albert' or args.graph_type == 'erdos_renyi' or args.graph_type == 'watts_strogatz':
        data_name = 'games'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        new_dataset = Games(path, n_graphs=args.n_graphs, n_games=args.n_games, n_nodes=args.n_nodes, m=args.m, 
                    target_spectral_radius=args.target_spectral_radius, alpha=args.alpha,
                    signal_to_noise_ratio=args.action_signal_to_noise_ratio, game_type=args.game_type,
                    regenerate_data=args.regenerate_data, graph_type=args.graph_type)
    else:
        raise NotImplementedError

    return new_dataset


def custom_collate(samples):
    length = []
    labels = []
    E = []
    U = []
   
    max_nodes = max([g['X'].shape[0] for g in samples])

    for i, g in enumerate(samples):
        X = g['X']
        A = g['A']
        A = torch.FloatTensor(A)   
        num_nodes = X.shape[0]
           
        XXT = torch.cov(X)    # [N, N]
        e, u = torch.linalg.eigh(XXT)  # [N], [N, N]
        e = torch.sort(e, dim=0, descending=False)[0]
        pad_e = e.new_zeros([max_nodes])   # padding eigenvalues with zero
        pad_e[:num_nodes] = e
        pad_u = u.new_zeros([max_nodes, max_nodes]) # padding eigenvectors with zero
        pad_u[:num_nodes, :num_nodes] = u

        E.append(pad_e)
        U.append(pad_u)
        length.append(num_nodes)

    E = torch.stack(E, 0)  # [B, N]
    U = torch.stack(U, 0)  # [B, N, N]
    length = torch.LongTensor(length)  # [B], number of nodes in each graph
    labels = torch.LongTensor(labels)  # [B]

    graphs = [samples[i]['g'] for i in range(len(samples))]
    edge_indices = [samples[i]['g'].edges() for i in range(len(samples))]

    batched_graph = dgl.batch(graphs) # batched graph
    # batched_graph = Batch.from_data_list(samples)
    #### MLP ###
    # sort_E = torch.stack([torch.sort(E[i,:], dim=0, descending=False)[0][-20:] for i in range(E.shape[0])], dim=0)
    # E = sort_E

    return E, U, batched_graph, length, edge_indices



def custom_train(samples):
    A_sets= []
    X_sets = []
    length = []
    priors = []

    max_nodes = max([g['X'].shape[0] for g in samples])

    for i, g in enumerate(samples):
        X = g['X']
        A = g['A']
        prior = g['prior']

        A = torch.FloatTensor(A)   
        num_nodes = X.shape[0]
           
        pad_X = X.new_zeros([max_nodes, X.shape[1]])   # padding eigenvalues with zero
        pad_X[:num_nodes, :X.shape[1]] = X

        pad_A = A.new_zeros([max_nodes, max_nodes]) # padding eigenvectors with zero
        pad_A[:num_nodes, :num_nodes] = A

        X_sets.append(pad_X)
        A_sets.append(pad_A)
        length.append(num_nodes)
        priors.append(prior)

    X_sets = torch.stack(X_sets, 0)  # [B, N]
    A_sets = torch.stack(A_sets, 0)  # [B, N, N]
    length = torch.LongTensor(length)  # [B], number of nodes in each graph
    priors = torch.stack(priors, 0)  # [B]

    return X_sets, A_sets, length, priors 

def to_device(x, device):
    if isinstance(x, list):
        return [to_device(a, device) for a in x]
    elif isinstance(x, tuple):
        return [to_device(a, device) for a in x]
    else:
        return x.to(device)

## get prior distribution for second-stage training, you can pre-train a interaction model and load it here.
def gated_prior(args, dataset, pre_model, device):
    data_loader = DataLoader(dataset,  batch_size = args.batch_size, collate_fn=custom_collate, shuffle = False)

    if args.pre_model_type == 'Specformer':
        pre_modelpath = f"../data/models/Specformer_random5000.pt"
        pre_model.load_state_dict(torch.load(pre_modelpath))
        pre_model = pre_model.to(device)
        pre_model.eval()       
        labels = []
        for ii, data in enumerate(data_loader):
            data = to_device(data, device)
            E, U, batch_graphs, length, edge_indices = data
            logits = pre_model(E, U, batch_graphs, length, edge_indices)
            y_pred = (torch.softmax(logits.squeeze(), dim = -1)>0.5).int()  # [B, num_classes]
            if E.shape[0] == 1:
                labels.append(y_pred[0].item())
            else:       
                for j in range(E.shape[0]):
                    labels.append(y_pred[j,0].item())    # [B]   prior id, if label = 1: correlation, 0: anticorrelation
        del pre_model
        torch.cuda.empty_cache()
    elif args.pre_model_type == 'MLP':
        pre_modelpath = f"../data/models/MLP_5000.pt"
        pre_model.load_state_dict(torch.load(pre_modelpath))
        pre_model = pre_model.to(device)
        pre_model.eval()
        labels = []    
        for i, data in enumerate(data_loader):
            E, _, _, _, _ = data 
            E = E.to(device)
            sort_E = torch.stack([torch.sort(E[k,:], dim=0, descending=False)[0][-args.n_eigen:] for k in range(E.shape[0])], dim=0)
            logits = pre_model(sort_E) 
            y_pred = (torch.softmax(logits.squeeze(), dim = -1)>0.5).int() # [B, num_classes] 
            if E.shape[0] == 1:
                labels.append(y_pred[0].item())
            else:
                for j in range(E.shape[0]):
                    labels.append(y_pred[j,0].item())  
        del pre_model
        torch.cuda.empty_cache() 
    elif args.pre_model_type == 'Random':   # Bernoulli(0.5)
        labels = []
        for j in range(len(dataset)):
            labels.append(2) 
    print(labels)
    new_dataset = []
    for i in range(len(dataset)):
        X = dataset[i]['X']
        X = torch.FloatTensor(X)
        if labels[i] == 1:
            if args.game_type == 'Yelp':
                prior = torch.cov(X)   # we use sparse prior for large graphs
                prior = prior - prior.min() 
                prior[prior > 1] = 1
                prior = torch.sqrt(prior)
            else:
                prior = (torch.corrcoef(X)+1)/2
                
        elif labels[i] == 0:
            if args.game_type == 'Yelp':
                prior = -1*torch.cov(X)
                prior = prior - prior.min() 
                prior[prior > 1] = 1
                prior = torch.sqrt(prior)
            else:
                prior = (-1*torch.corrcoef(X)+1)/2
            
        elif labels[i] == 2:
            prior = torch.zeros_like(dataset[i]['A']) + 0.5
    
        new_dataset.append({"X": X, "A": dataset[i]['A'], "prior": prior})

    return new_dataset