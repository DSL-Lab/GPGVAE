import argparse
import pickle
import time
import random
import string
import numpy as np
import wandb
import os
import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from ema_pytorch import EMA

from utils import get_encoder, get_decoder, mask_diagonal_and_sigmoid, reparameterization, gumbel_softmax, binary_concrete, kl_categorical_mc, nll_gaussian
from evaluation import eval_baseline, eval
from small_model import SpecformerSmall, MLPClassification
from get_dataset import data_preprocess, custom_collate
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# torch.set_default_dtype(torch.float64)
 
parser = argparse.ArgumentParser('Games')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--n_graphs', type=int, help='Number of graphs', default=1)
parser.add_argument('--n_nodes', type=int, help='Number of nodes', default=20)
parser.add_argument('--n_games', type=int, help='Number of games', default=100)  
parser.add_argument('--num_mix_component', type=int, help='Number of mix component', default=10)
parser.add_argument('--sparsity', type=float, help='ratio of zeros', default=0.8)
parser.add_argument('--alpha', type=float, help='Smoothness of marginal benefits', default=1)
parser.add_argument('--target_spectral_radius', type=float, help='Target spectral radius', default=0.4)
parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=600)
parser.add_argument('--patience', type=int, help='Early Stopping Patience', default=50)
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
parser.add_argument('--weight_decay', type=float, help='decay parameter in AdamW ', default=1e-4)
parser.add_argument('--warm_up_epoch', type=int, help='Number of lr warm_up', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=100)
parser.add_argument('--n_eigen', type=int, help='number of input eigenvalues.', default=20)
parser.add_argument('--n_class', type=int, help='Number of classes', default=2)
parser.add_argument('--n_layer', type=int, help='Number of layers in specformer', default=4)
parser.add_argument('--n_heads', type=int, help='Number of heads', default=4)
parser.add_argument('--n_gcn', type=int, help='Number of layers in GCN', default=2)
parser.add_argument('--feat_dropout', type=float, help='Feature dropout', default=0.1)
parser.add_argument('--trans_dropout', type=float, help='Transformer dropout', default=0.1)
parser.add_argument('--adj_dropout', type=float, help='Adjacency dropout', default=0.3)
parser.add_argument('--hidden_dim', type=int, help='Dimension of node embeddings', default=64)
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', default=True, help='Uses discrete samples in training forward pass.')
parser.add_argument('--encoderA', type=str, help='Types of encoder', default="per_game_transformer", choices=["mlp_on_seq", "per_game_transformer", "transformer"])
parser.add_argument('--encoderZ', type=str, help='Types of encoder', default="mlp_on_seq", choices=["mlp_on_nodes", "mlp_on_seq"])
parser.add_argument('--regenerate_data', action='store_true', help='Whether to regenerate the graphs')
parser.add_argument('--noise_std', type=float, help='B noise std.', default=0.)
parser.add_argument('--m', type=int, help='Barabasi-Albert parameter m', default=3)
parser.add_argument('--action_signal_to_noise_ratio', type=float, help='Signal-to-noise ration in synthetic actions', default=10)
parser.add_argument('--graph_type', type=str, help='Type of graph', default="barabasi_albert", choices=["barabasi_albert", "erdos_renyi", "watts_strogatz", "indian_village", "yelp_rating", 'yelp_review', "NYC", "TKY", "IST","SaoPaulo","Jakarta","KualaLampur"])
parser.add_argument('--game_type', type=str, help='Type of game', default="linear_quadratic", choices=["linear_quadratic", "linear_influence", "barik_honorio", "realworld"])
parser.add_argument('--yelp_dump_filename', type=str, help='Name of the file with the Yelp dataset.', default="PA_dataset_foodcategory_review")
parser.add_argument('--cost_distribution', type=str, help='Type of distribution to use to sample node-wise costs.', default="normal", choices=["normal", "uniform"])
parser.add_argument('--pre_model_type', type = str, help='Type of model', default="MLP", choices=["MLP", "Specformer", "Random"])



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


class Model(torch.nn.Module):
    def __init__(self, encoderA_type, encoderZ_type, temp, hard, n_nodes, n_games, hidden_dim, n_layer, num_mix_component):
        super(Model, self).__init__()

        self.hard = hard
        self.temp = temp
        self.n_nodes = n_nodes
        self.num_mix_component = num_mix_component
        self.encoder_A = get_encoder(encoder_type=encoderA_type, n_nodes=n_nodes, n_games=n_games,
                            hidden_dim=hidden_dim, num_mix_component = num_mix_component)
        self.encoder_Z = get_encoder(encoder_type=encoderZ_type, n_nodes=n_nodes, n_games=n_games,
                            hidden_dim=hidden_dim, num_mix_component = num_mix_component)
        
        self.decoder = get_decoder(hidden_dim=hidden_dim, out_channels = n_games, n_layer = n_layer)

    def forward(self, X):
        device = X.device
        Z_mu = self.encoder_Z(X) # B X N X H
        Z_logvar = torch.tensor(-3.0)
         
        logit_theta, logit_alpha = self.encoder_A(X) #  logit_theta:  B X N(N+1)/2 X K, logit_alpha: B X K, logit_prior: N X N X 3, laplace_mu: N X N
        n_graphs, n_nodes = X.size(0), X.size(1) # B, N
        prob_theta = torch.stack([mask_diagonal_and_sigmoid(logit_theta[:,:,:,kk]) for kk in range(self.num_mix_component)], dim = -1)     
        logit_theta = logit_theta.reshape(n_graphs, -1, self.num_mix_component)
       
        N_samples = 100
        loss_nll = 0
        is_sym = True
        for _ in range(N_samples):
            sample_Z = reparameterization(Z_mu, Z_logvar) # node latent variables  B X N X H 
            sample_alpha = gumbel_softmax(logit_alpha, tau=self.temp, hard=True) # B X K  use hard sample trick to get one discrete category for mixture components
            logit_prob = torch.stack([torch.mm(logit_theta[num], sample_alpha[num].reshape(self.num_mix_component, 1)) for num in range(n_graphs)], dim = 0)
            logit_prob = logit_prob.reshape(n_graphs, n_nodes, n_nodes)
            sample_A = binary_concrete(logit_prob, tau=self.temp, hard=True)  # B X N X N, use hard sample trick to get discrete sampling
            
            if is_sym:
                sample_A = torch.tril(sample_A, diagonal=-1)
                sample_A = sample_A + sample_A.transpose(1, 2)    

            Xmu = self.decoder(sample_Z.to(device), sample_A.to(device))     # B X N X G  G: n_games
            loss_nll += nll_gaussian(Xmu, X)/N_samples
         
        return prob_theta, logit_alpha, Z_mu, loss_nll
    

def run(args):
    # wandb.init(project="SSVAE", entity="xueyu", config=args)
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
    model_path = f"../data/models/{model_name}.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.game_type == 'linear_quadratic':
        project_dir = 'output_synthetic'
        output_dir = os.path.join(project_dir, 'results_'+str(args.pre_model_type)+str(args.graph_type)+str(args.game_type)+'_alpha'+str(args.alpha)+'_beta'+str(args.target_spectral_radius)+'.pickle') 
    elif args.game_type == 'linear_influence':
        project_dir = 'output_synthetic'
        output_dir = os.path.join(project_dir, 'results_'+str(args.pre_model_type)+str(args.graph_type)+str(args.game_type)+'_alpha'+str(args.alpha)+'.pickle') 
    elif args.game_type == 'barik_honorio':
        project_dir = 'output_synthetic'
        output_dir = os.path.join(project_dir, 'results_'+str(args.pre_model_type)+str(args.graph_type)+str(args.game_type)+'.pickle') 
    else:
        project_dir = 'output_realworld'
        output_dir = os.path.join(project_dir, 'results_'+str(args.pre_model_type)+str(args.graph_type)+'.pickle') 

    # negative_game = {}
    # for alpha in [0.9, 0.7, 0.5, 0.3]:
    #     args.alpha = alpha
    #     args.target_spectral_radius = -0.8
    #     dataset = data_preprocess(args)
    #     X = dataset[0]['X']
    #     # XXT = torch.matmul(X, X.T) / X.shape[1]
    #     XXT = torch.cov(X)
    #     e_val = torch.linalg.eigvals(XXT).real
    #     negative_game[f'alpha={alpha}'] = sorted(e_val.numpy().tolist())
    
    # positive_game = {}
    # for alpha in [0.9, 0.7, 0.5, 0.3]:
    #     args.alpha = alpha
    #     args.target_spectral_radius = 0.8
    #     dataset = data_preprocess(args)
    #     X = dataset[0]['X']
    #     # XXT = torch.matmul(X, X.T)/ X.shape[1]
    #     XXT = torch.cov(X)
    #     e_val = torch.linalg.eigvals(XXT).real
    #     positive_game[f'alpha={alpha}'] = sorted(e_val.numpy().tolist())
    # plot_side_by_side(positive_game, negative_game)
    # plt.savefig(f'../figures/{args.graph_type}_alpha_postive_vs_negative.png', bbox_inches='tight')

    # # eigen ratio wrt se
    # plt.figure()
    # args.n_graph = 20
    # betas = list(np.arange(-0.95, 0.95, 0.05))
    # max_eig = []
    # for beta in betas:
    #     args.alpha = 0.5
    #     args.target_spectral_radius = beta
    #     dataset = data_preprocess(args)
    #     max_values = []
    #     for i in range(args.n_graph):
    #         X = dataset[i]['X']
    #         # XXT = torch.matmul(X, X.T) / X.shape[1]
    #         XXT = torch.cov(X)
    #         e_vals = torch.linalg.eigvals(XXT).real.numpy().tolist()
    #         max_values.append(max(e_vals)/sum(e_vals))
    #     max_eig.append(np.mean(max_values))
    # plot_sinlge(betas, max_eig, "", "Beta", "Max Eigen Value Ratio")
    # plt.savefig(f'../figures/{args.graph_type}_beta_max_eigvalue.png', bbox_inches='tight')
    
    # # change wrt to K
    # plt.figure()
    # args.alpha = 0.5
    # max_eig = []
    # for n_game in range(5, 200):
    #     args.n_games = n_game
    #     dataset = data_preprocess(args)
    #     X = dataset[0]['X']
    #     XXT = torch.matmul(X, X.T) / X.shape[1]
    #     e_vals = torch.linalg.eigvals(XXT).real.numpy().tolist()
    #     max_eig.append(max(e_vals)/sum(e_vals))
    # plot_sinlge(list(range(5, 200)), max_eig, "", "Number of Games", "Max Eigen Value Ratio")
    # plt.savefig(f'./figures/{args.graph_type}_k_max_eigvalue.png', bbox_inches='tight')
    colors = ['tab:green', 'tab:blue', 'r', 'darkorange', 'tab:brown', 'm', 'c','purple','yellow']

    # plt.figure(figsize=(5, 5))
    # plt.subplot(1, 2, 1)
    # eigen ratio wrt se linear influence games
    args.n_graph = 20
    # alphas = [0, 0.5, 0.9]
    
    # for a in range(len(alphas)):
    #     args.alpha = alphas[a]
    #     args.game_type = 'linear_influence'
    #     dataset = data_preprocess(args)
    #     X = dataset[0]['X']
    #     XXT = torch.cov(X)
    #     e_vals = torch.linalg.eigvals(XXT).real.numpy().tolist()
    #     plt.plot(sorted(e_vals, reverse=False), color=colors[a], label=r'$\alpha=$'+str(alphas[a]))

    # plt.ylabel(r'$\lambda$', fontsize=12)
    # plt.xlabel(r'$\lambda$ Index', fontsize=12)
    # # plt.title('Linear Influence Games', fontsize=12)
    # plt.legend(loc='upper left', fontsize=8)

    fig = plt.figure(figsize=(5, 5))
    alpha = np.arange(0, 0.9, 0.05)
    beta = np.arange(-0.9, 0.9, 0.1)
    abc = []
    args.game_type = 'linear_quadratic'
    graph_type = ['erdos_renyi']
    for g in range(len(graph_type)):
        args.graph_type = graph_type[g]
        ax = fig.add_subplot(1, 1, g+1, projection='3d')
        for a in alpha:
            for b in beta:
                args.alpha = a
                args.target_spectral_radius = b
                dataset = data_preprocess(args)
                X = dataset[0]['X']
            # XXT = torch.matmul(X, X.T) / X.shape[1]
                XXT = torch.cov(X)
                e_vals = torch.linalg.eigvals(XXT).real.numpy().tolist()
                c = max(e_vals)/sum(e_vals)
                abc.append((a, b, c))
        plot_3d(abc, ax, graph_type[g])
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.23)   
    plt.savefig(f'../figures/eigen_3dplot.pdf', bbox_inches='tight')


def plot_3d(data, ax, graph_type):
    from scipy.interpolate import griddata
    a = [x[0] for x in data]
    b = [x[1] for x in data]
    c = [x[2] for x in data]
    # Generate a grid over the data and interpolate the c values for a smooth surface
    xi = np.linspace(min(a), max(a), num=200)
    yi = np.linspace(min(b), max(b), num=200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((a, b), c, (xi, yi), method='cubic')

    # Set Seaborn style
    sns.set(style="white")

    # Create the 3D surface plot
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='k', alpha=0.7)
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='k', alpha=0.7)
    colorbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    colorbar.ax.tick_params(labelsize=10)

    # Customize the plot
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'Spectral radius $\rho(\beta \mathbf{A})$', fontsize=12)
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r'$\frac{\lambda_{\max}}{\sum{\lambda_i}}$', fontsize=12, rotation=0)

    # zlabel.set_rotation(45)  # Rotate the z-label by 90 degrees
    if graph_type == 'erdos_renyi':
        ax.set_title('ER graphs, Linear quadratic games', fontsize=12)
    else:
        ax.set_title('WS graphs', fontsize=12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([-0.9, -0.4, 0, 0.4, 0.9])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=8)
    ax.set_yticklabels([-0.9, -0.4, 0, 0.4, 0.9], fontsize=8)
    ax.set_zticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax.azim, ax.elev = (-155, 30)
    # Show the plot
    # plt.show()

def plot_sinlge(x, y, title, x_label, y_label):
    sns.set_style("darkgrid")
    sns.lineplot(x=x, y=y, marker='o')

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)


def plot_side_by_side(
        data_group_1,
        data_group_2,
        title1='Positive Game',
        title2='Negative Game',
        y_label=r'$\lambda$',
        x_label=r'$\lambda$ Index',
        x_len=20,
    ):
    """
    Plots two groups of line plots side by side with shared X and Y scales.
    
    :param data_group_1: A dictionary where keys are labels and values are lists 
                         of y-coordinates for the first group.
    :param data_group_2: A dictionary where keys are labels and values are lists 
                         of y-coordinates for the second group.
    """
    from matplotlib.ticker import MaxNLocator
    sns.set_style("darkgrid")
    # Initialize the subplots with shared x and y axes
    fig, axs = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)
    
    axs[0].set_title(title1)
    axs[1].set_title(title2)
    
    # Plot the lines for the first group
    for label, y_data in data_group_1.items():
        x_data = list(range(len(y_data)))
        sns.lineplot(x=x_data, y=y_data, label=label, ax=axs[0], marker='o')
    
    # Plot the lines for the second group
    for label, y_data in data_group_2.items():
        x_data = list(range(len(y_data)))
        sns.lineplot(x=x_data, y=y_data, label=label, ax=axs[1], marker='o')

    # Set the labels and title of the shared axes
    # axs[0].set_xlabel('Shared X Axis Label')
    axs[0].set_ylabel(y_label)
    axs[0].set_xlabel(x_label)
    axs[1].set_xlabel(x_label)

    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_xticks(range(1, x_len+1))
    axs[1].set_xticks(range(1, x_len+1))
    
    plt.tight_layout()
    plt.legend(loc='upper left')
    # plt.show()


def plot_multiple_lines(data_dict):
    """
    Plots multiple lines on the same plot, each with a unique label.
    
    :param data_dict: A dictionary where keys are labels and values are lists 
                      of y-coordinates. The index of the list represents the 
                      x-coordinate.
    """
    
    # Set style
    sns.set_style("darkgrid")
    
    # Initialize the plot
    plt.figure(figsize=(10,6))
    
    # Loop over the data_dict and plot each line with a label
    for label, y_data in data_dict.items():
        x_data = list(range(len(y_data)))  # Assume x is the index of the y_data
        sns.lineplot(x=x_data, y=y_data, label=label)
        
    # Customize the axes and show the plot
    plt.xlabel('')
    plt.ylabel(r'$\lambda$')
    plt.title('Place Holder')
    plt.legend(loc='upper left')
    plt.show()
    
if __name__ == "__main__":
    args = parser.parse_args()
    
    print(f"graph:{args.graph_type} game:{args.game_type} alpha: {args.alpha}")  
    run(args)
             
