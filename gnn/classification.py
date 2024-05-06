import os.path as osp
import os
import argparse
import time
import pickle
import math
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from utils import compute_roc_auc_score
from torch.utils.data import DataLoader

import dgl
from games_dataset import Pretrain_Games
from small_model import MLPClassification, SpecformerSmall
import wandb
import warnings
warnings.filterwarnings("ignore")
# torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser('Games')
parser.add_argument('--seed', type=int, help='Random seed.', default=12)
parser.add_argument('--n_graphs', type=int, help='Number of graphs', default=2000)
parser.add_argument('--n_games', type=int, help='Number of games', default=100)
parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=50)
parser.add_argument('--patience', type=int, help='Patience', default=50)
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
parser.add_argument('--weight_decay', type=float, help='decay parameter in AdamW ', default=1e-4)
parser.add_argument('--warm_up_epoch', type=int, help='Number of lr warm_up', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--n_eigen', type=int, help='number of input eigenvalues.', default=20)
parser.add_argument('--n_class', type=int, help='Number of classes', default=2)
parser.add_argument('--n_layer', type=int, help='Number of layers', default=4)
parser.add_argument('--n_heads', type=int, help='Number of heads', default=4)
parser.add_argument('--feat_dropout', type=float, help='Feature dropout', default=0.1)
parser.add_argument('--trans_dropout', type=float, help='Transformer dropout', default=0.1)
parser.add_argument('--adj_dropout', type=float, help='Adjacency dropout', default=0.1)
parser.add_argument('--hidden_dim', type=int, help='Dimension of node embeddings', default=64)
parser.add_argument('--regenerate_data', action='store_true', help='Whether to regenerate the graphs')
parser.add_argument('--noise_std', type=float, help='B noise std.', default=0.)
parser.add_argument('--m', type=int, help='Barabasi-Albert parameter m', default=3)
parser.add_argument('--graph_type', type=str, help='Type of graph', default="barabasi_albert", choices=["barabasi_albert", "erdos_renyi", "watts_strogatz", "indian_village", "yelp", "NYC", "TKY", "IST","SaoPaulo","Jakarta","KualaLampur"])
parser.add_argument('--game_type', type=str, help='Type of game', default="linear_quadratic", choices=["linear_quadratic", "linear_influence", "barik_honorio", "realworld"])

parser.add_argument('--action_signal_to_noise_ratio', type=float, help='Signal-to-noise ration in synthetic actions', default=10)
parser.add_argument('--cost_distribution', type=str, help='Type of distribution to use to sample node-wise costs.', default="normal", choices=["normal", "uniform"])
parser.add_argument('--model_type', type = str, help='Type of model', default="Specformer", choices=["MLP", "Specformer"])


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def correlation_baseline_score(x, A_true):
        X_mean = torch.mean(x, dim=-1, keepdim=True)
        X_shift = x - X_mean
        cov = X_shift @ X_shift.transpose(0, 1)
        X_var = (X_shift * X_shift).sum(dim=-1, keepdim=True)
        corr_coeff = cov / (torch.sqrt(X_var * X_var.transpose(0, 1))+1e-8)
        A_pred = torch.sigmoid(torch.Tensor(corr_coeff))
        A_pred.fill_diagonal_(0.)   
        roc_score = compute_roc_auc_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))

        anti_A_pred = torch.sigmoid(torch.Tensor(-1*corr_coeff))
        anti_A_pred.fill_diagonal_(0.)  
        anti_roc_score = compute_roc_auc_score(A_true.unsqueeze(0), anti_A_pred.unsqueeze(0))
        return roc_score, anti_roc_score


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

        corr_roc, anticorr_roc = correlation_baseline_score(X, A)
        if corr_roc >= anticorr_roc:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
           
        XXT = torch.cov(X)    # [N, N]
        e, u = torch.linalg.eigh(XXT)
        e = torch.sort(e, dim=0, descending=False)[0]
        pad_e = e.new_zeros([max_nodes])   # padding eigenvalues with zero
        pad_e[:num_nodes] = e
        # pad_e[:num_nodes-1] = torch.FloatTensor(differences)


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
    #### if we use MLP ###
    # sort_E = torch.stack([torch.sort(E[i,:], dim=0, descending=False)[0][-20:] for i in range(E.shape[0])], dim=0)
    # E = sort_E

    return E, U, batched_graph, length, labels, edge_indices


def run(args):
    # wandb.init(project="SSVAE", entity="xueyu", config=args)
    model_path = f"../data/models/{args.model_type}.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'games'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)  
    dataset = Pretrain_Games(path, n_graphs=args.n_graphs, n_games = args.n_games, m=args.m, signal_to_noise_ratio=args.action_signal_to_noise_ratio,
                    regenerate_data=args.regenerate_data, cost_distribution=args.cost_distribution)
    # test_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'test_games') 
    # test = Games(test_path, n_graphs=100, n_games=args.n_games, n_nodes=30, m=args.m, 
    #                 target_spectral_radius=0.2, alpha=0, signal_to_noise_ratio=args.action_signal_to_noise_ratio, game_type=args.game_type,
    #                 regenerate_data=args.regenerate_data, graph_type=args.graph_type,  cost_distribution=args.cost_distribution)
    
    train, valid, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.1*len(dataset)), len(dataset)-int(0.9*len(dataset))]) 
    train_dataloader = DataLoader(train, batch_size = args.batch_size, collate_fn=custom_collate, shuffle = True)
    valid_dataloader = DataLoader(valid, batch_size = args.batch_size // 2, collate_fn=custom_collate, shuffle = False)
    test_dataloader  = DataLoader(test,  batch_size = args.batch_size // 2, collate_fn=custom_collate, shuffle = False)

    models_results = {}
    if args.model_type == 'MLP':
        model = MLPClassification(n_eigen=args.n_eigen, hidden_dim=args.hidden_dim).to(device)
    elif args.model_type == 'Specformer':
        model = SpecformerSmall(args.n_class, args.n_layer, args.hidden_dim, args.n_heads, args.feat_dropout, args.trans_dropout, args.adj_dropout).to(device)

    print(count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # warm_up + cosine weight decay
    lr_plan = lambda cur_epoch: (cur_epoch+1) / args.warm_up_epoch if cur_epoch < args.warm_up_epoch else \
              (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - args.warm_up_epoch) / (args.n_epochs - args.warm_up_epoch))))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)

    loss_fn = torch.nn.BCELoss()
    print("Starting training")
    start = time.time()    

    loss_all = []
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []

    model.train()
    # # wandb.watch(model, log="all", log_freq=10)

    for epoch in range(args.n_epochs):
        epoch_loss = 0

        for data in train_dataloader:
            e, u, g, length, y, edge_indices = data   # eigenvalue, eigenvector, X, A, label
            e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)

            if args.model_type == 'Specformer':
                logits = model(e, u, g, length, edge_indices)  # [B, num_classes]
            elif args.model_type == 'MLP':
                logits = model(e) 

            optimizer.zero_grad()
            y_pred = torch.softmax(logits.squeeze(), dim = -1)
            loss = loss_fn(y_pred.to(torch.float32), y.to(torch.float32))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
        scheduler.step()
        
        epoch_loss /= len(train_dataloader)
        train_acc = binary_accuracy(model, train_dataloader, model_type=args.model_type)
        val_acc = binary_accuracy(model, valid_dataloader, model_type=args.model_type)
        test_acc = binary_accuracy(model, test_dataloader, model_type=args.model_type)
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch} --loss={epoch_loss:.4f} --train_acc = {train_acc:.4f}  --val_accuracy={val_acc:.4f} --test_acc = {test_acc:.4f}')
        
        if epoch == 0 or val_acc >= max(val_accuracy):
            torch.save(model.state_dict(), model_path) 

        loss_all.append(epoch_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        test_accuracy.append(test_acc)

        if epoch > args.patience and max(val_accuracy[-args.patience:]) < max(val_accuracy):
            print("Early stopping")
            break
        # wandb.log({'train loss':epoch_loss, 'val': val_acc, 'test': test_acc})

    print("Training finished in {:.4f}s".format(time.time() - start))
    # wandb.finish()

    models_results["test_acc"] = test_accuracy
    models_results["train_acc"] = train_accuracy
    models_results["loss"] = loss_all
    with open('results_'+str(args.model_type)+'.pickle', 'wb') as outfile:
        pickle.dump(models_results, outfile)
    print(f'results saved in results_'+str(args.model_type)+'.pickle')



def binary_accuracy(model, data_loader, model_type):
    model.eval()  # Set the model to evaluation mode, very important!!!
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            e, u, g, length, y, edge_indices = data   # eigenvalue, eigenvector, graph, length, label
            e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)
            if model_type == 'Specformer':
                logits = model(e, u, g, length, edge_indices)  # [B, num_classes]
            elif model_type == 'MLP':
                logits = model(e) 
            y_pred = (torch.softmax(logits.squeeze(), dim = -1)>0.5).int()
            total += y.size(0)
            correct += (y_pred[:,0] == y[:,0]).sum().item()

    accuracy = (correct / total)
    return accuracy


if __name__ == "__main__":
    ## pre-training
    args = parser.parse_args()
    run(args)
    

   



