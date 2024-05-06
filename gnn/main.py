import argparse
import pickle
import time
import random
import string
import numpy as np
import wandb
import os
import concurrent.futures

import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from utils import get_encoder, get_decoder, mask_diagonal_and_sigmoid, reparameterization, gumbel_softmax, binary_concrete, nll_gaussian, kl_categorical_mc, kl_categorical_approx
from evaluation import eval_baseline, eval, compute_validloss
from small_model import SpecformerSmall, MLPClassification
from get_dataset import data_preprocess, gated_prior, custom_train
import warnings
warnings.filterwarnings("ignore")
# torch.set_default_dtype(torch.float64)


parser = argparse.ArgumentParser('Games')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--n_graphs', type=int, help='Number of graphs', default=1)
parser.add_argument('--n_nodes', type=int, help='Number of nodes', default=20)
parser.add_argument('--n_games', type=int, help='Number of games', default=100)  
parser.add_argument('--num_mix_component', type=int, help='Number of mix component', default=5)
parser.add_argument('--alpha', type=float, help='Smoothness of marginal benefits', default=0.5)
parser.add_argument('--target_spectral_radius', type=float, help='Target spectral radius', default=0.4)
parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=200)
parser.add_argument('--patience', type=int, help='Early Stopping Patience', default=50)
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
parser.add_argument('--weight_decay', type=float, help='decay parameter in AdamW ', default=1e-5)
parser.add_argument('--warm_up_epoch', type=int, help='Number of lr warm_up', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
parser.add_argument('--n_heads', type=int, help='Number of heads in encoder', default=4)
parser.add_argument('--n_eigen', type=int, help='number of input eigenvalues.', default=20)
parser.add_argument('--n_class', type=int, help='Number of classes', default=2)
parser.add_argument('--n_layer', type=int, help='Number of layers in specformer', default=4)
parser.add_argument('--n_preheads', type=int, help='Number of heads in specformer', default=4)
parser.add_argument('--n_gcn', type=int, help='Number of layers in GCN', default=2)
parser.add_argument('--feat_dropout', type=float, help='Feature dropout', default=0.1)
parser.add_argument('--trans_dropout', type=float, help='Transformer dropout', default=0.1)
parser.add_argument('--adj_dropout', type=float, help='Adjacency dropout', default=0.3)
parser.add_argument('--hidden_dim_1st', type=int, help='Dimension of embeddings in first stage', default=64)
parser.add_argument('--hidden_dim', type=int, help='Dimension of embeddings in second stage', default=256)
parser.add_argument('--beta0', type=float, help='control weight of recon', default=1)
parser.add_argument('--beta1', type=float, help='control weight of kl_A', default=10)
parser.add_argument('--beta2', type=float, help='control weight of kl_A', default=1)
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', default=True, help='Uses discrete samples in training forward pass.')
parser.add_argument('--encoderA', type=str, help='Types of encoder', default="transformer", choices=["per_game_transformer", "transformer", "mlp_encoder_2"])
parser.add_argument('--encoderZ', type=str, help='Types of encoder', default="gcn_on_z", choices=["gcn_on_z", "mlp_on_seq", 'transformer_z'])
parser.add_argument('--regenerate_data', action='store_true', help='Whether to regenerate the graphs')
parser.add_argument('--noise_std', type=float, help='B noise std.', default=0.)
parser.add_argument('--m', type=int, help='Barabasi-Albert parameter m', default=3)
parser.add_argument('--action_signal_to_noise_ratio', type=float, help='Signal-to-noise ration in synthetic actions', default=10)
parser.add_argument('--graph_type', type=str, help='Type of graph', default="indian_village", choices=["stochastic_block_model", "barabasi_albert", "erdos_renyi", "watts_strogatz", "indian_village", "PA_review", 'PA_rating', "LA_rating", 'LA_review', "NYC", "TKY", "IST","SaoPaulo","Jakarta","KualaLampur"])
parser.add_argument('--game_type', type=str, help='Type of game', default="village", choices=["linear_quadratic", "linear_influence", "barik_honorio", "Yelp", "Foursquare", "village"])
parser.add_argument('--cost_distribution', type=str, help='Type of distribution to use to sample node-wise costs.', default="normal", choices=["normal", "uniform"])
parser.add_argument('--pre_model_type', type = str, help='Type of model', default="Specformer", choices=["Specformer", "MLP", "Random"])
parser.add_argument('--use_amp', action='store_true', help='Whether to use automatic mixed precision')
parser.add_argument('--estimator', type=str, help='Types of estimator', default="concrete", choices=["soft_approx", "concrete", "reinforce", 'measure'])
parser.add_argument('--eta', type=float, help='control weight of reinforce', default=0.99)
parser.add_argument('--klA_nsample', type=int, help='Number of samples for KL A', default=50)
parser.add_argument('--nll_nsample', type=int, help='Number of samples for reconstruction loss', default=1)


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


class Encoding(torch.nn.Module):
    def __init__(self, encoderA_type, encoderZ_type, n_games, n_heads, n_nodes, hidden_dim, num_mix_component):
        super(Encoding, self).__init__()
        self.num_mix_component = num_mix_component
        self.encoder_A = get_encoder(encoder_type=encoderA_type, n_games=n_games, n_heads=n_heads, n_nodes = n_nodes,
                            hidden_dim=hidden_dim, num_mix_component = num_mix_component)
        self.encoder_Z = get_encoder(encoder_type=encoderZ_type, n_games=n_games, n_heads=n_heads, n_nodes = n_nodes,
                            hidden_dim=hidden_dim, num_mix_component = num_mix_component)
        
    def forward(self, X):
        Z_mu = self.encoder_Z(X) # B X N X H
        Z_logstd = torch.tensor(0.0)     
        logit_theta, logit_alpha = self.encoder_A(X) #  logit_theta:  B X N X N X K, logit_alpha: B X K, logit_prior: N X N X 3, laplace_mu: N X N       
        return logit_theta, logit_alpha, Z_mu, Z_logstd
    

class Decoding(torch.nn.Module):
    def __init__(self, n_games, hidden_dim, n_layer, num_mix_component):
        super(Decoding, self).__init__()
        self.num_mix_component = num_mix_component
        self.decoder = get_decoder(hidden_dim=hidden_dim, out_channels = n_games, n_layer = n_layer)

    def forward(self, sample_z, sample_A):
        device = sample_z.device
        Xmu = self.decoder(sample_z.to(device), sample_A.to(device))     # B X N X G  G: n_games 
        X_logstd = torch.tensor(0.0)
        return Xmu, X_logstd
     

def concrete_mc(param):
    X, Z_mu, Z_logstd, logit_alpha, logit_theta, Decoder, args, ii = param
    device = Z_mu.device
    sample_Z = reparameterization(Z_mu, Z_logstd) # node latent variables  B X N X H, B=1
    sample_alpha = gumbel_softmax(logit_alpha, tau=args.temp, hard=True) # B X K  use hard sample trick to get one discrete category for mixture components
    logit_adj = logit_theta @ sample_alpha.reshape(args.num_mix_component, 1) # B X N X N X 1
    prob_adj = binary_concrete(logit_adj.squeeze(-1), tau=args.temp, hard=False)  # B X N X N, use hard sample trick to get discrete sampling
    mask = ~torch.eye(prob_adj.shape[1], dtype=bool, device=prob_adj.device).repeat(prob_adj.shape[0],1,1)
    prob_adj = prob_adj * mask
    Xmu, X_logstd = Decoder(sample_Z.to(device), prob_adj.to(device))     # B X N X G  G: n_games
    loss_nll = nll_gaussian(Xmu, X, X_logstd)
    return loss_nll

def reinforce_mc(param):
    X, Z_mu, Z_logstd, logit_alpha, prob_theta, Decoder, args, ii = param
    device = Z_mu.device
    sample_Z = reparameterization(Z_mu, Z_logstd) # node latent variables  B X N X H 
    s_alpha = torch.multinomial(torch.softmax(logit_alpha, -1), num_samples = 1, replacement=True) # B X 1
    prob_adj = prob_theta[:, :, :, s_alpha.item()] # B X N X N
    s_adj = torch.bernoulli(prob_adj) # B X N X N
    # The Reinforce loss is just log_prob*loss
    eps = 1e-16
    adj_loss = torch.stack([(s_adj * (torch.log(prob_theta[0,:,:, kk] + eps)) + (1-s_adj) * (torch.log((1-prob_theta[0,:,:,kk]) + eps))) for kk in range(args.num_mix_component)], dim=1)  # B X K X N X N
    adj_loss = adj_loss.reshape(1, args.num_mix_component, args.n_nodes**2)
    adj_loss = torch.sum(adj_loss, dim = -1)/(args.n_nodes**2)  # B X K
    log_alpha = torch.log_softmax(logit_alpha, -1)  # B X K
    log_prob = adj_loss + log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1) # B X 1

    mask = ~torch.eye(prob_adj.shape[1], dtype=bool, device=prob_adj.device).repeat(prob_adj.shape[0],1,1)
    s_adj = s_adj * mask
    Xmu, X_logstd = Decoder(sample_Z.to(device), s_adj.to(device))     # B X N X G  G: n_games
    nll = nll_gaussian(Xmu, X, X_logstd)

    reinforce_loss = torch.mean(log_prob*nll.detach())
    return nll, reinforce_loss


def sample_loss(args):
    Decoder, idd, sample_Z, prob_theta, alpha_k, X = args
    device = sample_Z.device
    loss_positive = 0
    loss_negative = 0
    ii, jj, kk = idd
    adj_k = torch.bernoulli(prob_theta) # B X N X N
    adj_k[:, ii, jj] = 1
    Xmu, X_logstd = Decoder(sample_Z, adj_k.to(device))
    loss_positive = nll_gaussian(Xmu, X, X_logstd)
    adj_k = torch.bernoulli(prob_theta) # B X N X N
    adj_k[:, ii, jj] = 0
    Xmu, X_logstd = Decoder(sample_Z.to(device), adj_k.to(device))
    loss_negative = nll_gaussian(Xmu, X, X_logstd)
    thetaijk_grad = alpha_k * (loss_positive.detach() - loss_negative.detach())
    return thetaijk_grad, idd

def run(args):
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
    model_path = f"../data/models/{model_name}.pt"
    device ='cuda' if torch.cuda.is_available() else 'cpu'

    dataset = data_preprocess(args)
    max_nodes = 0
    for i in range(len(dataset)):
        if dataset[i]['X'].shape[0] > max_nodes:
            max_nodes = dataset[i]['X'].shape[0] 
    args.n_nodes = max_nodes
    args.n_games = dataset[0]['X'].shape[1]

    ###########  load pre-trained model  #############
    if args.pre_model_type == 'MLP':
        pre_model = MLPClassification(n_eigen=args.n_eigen, hidden_dim=args.hidden_dim_1st).to(device)   
    elif args.pre_model_type == 'Specformer':
        pre_model = SpecformerSmall(args.n_class, args.n_layer, args.hidden_dim_1st, args.n_preheads, args.feat_dropout, args.trans_dropout, args.adj_dropout).to(device)

    new_dataset = gated_prior(args, dataset, pre_model, device)
    train_loader = DataLoader(new_dataset, batch_size = args.batch_size, collate_fn=custom_train, shuffle = True) # X_sets, A_sets, labels
    
    Encoder = Encoding(encoderA_type = args.encoderA, encoderZ_type=args.encoderZ, n_games=args.n_games, n_heads=args.n_heads, n_nodes=args.n_nodes, hidden_dim=args.hidden_dim, num_mix_component=args.num_mix_component).to(device)
    Decoder = Decoding(n_games=args.n_games, hidden_dim=args.hidden_dim, n_layer=args.n_layer, num_mix_component=args.num_mix_component).to(device)
    model = torch.nn.Sequential(Encoder, Decoder)
    print(f"Number of parameters: {count_parameters(model)}")
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    lr_plan = lambda cur_epoch: (cur_epoch+1) / args.warm_up_epoch if cur_epoch < args.warm_up_epoch else \
            (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - args.warm_up_epoch) / (args.n_epochs - args.warm_up_epoch))))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)

    print("Starting training")
    loss_negELBO = []
    train_roc_aucs = []
    old_null = 0
    model.train()
    print(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    # wandb.init(project="GPGVAE", entity="xueyu", config=args)
    # wandb.watch(model, log="all", log_freq=10)

    for epoch in range(args.n_epochs):
        start = time.time()    
        beta0 = args.beta0         
        beta1 = args.beta1  # KL A
        beta2 = args.beta2  # KL Z 
        epoch_loss = 0
        train_REloss = 0
        train_KLAloss = 0
        train_KLZloss = 0
        for data in train_loader:
            optimizer.zero_grad()
            X, _, _, prior_i = data     # B X N X G, B X N X N, B
            X, prior_i = X.to(device), prior_i.to(device)    # B X N X G, B

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.use_amp):
                logit_theta, logit_alpha, Z_mu, Z_logstd = Encoder(X)
                prob_theta = torch.stack([mask_diagonal_and_sigmoid(logit_theta[:,:,:,kk]) for kk in range(args.num_mix_component)], dim = -1) 
                if args.estimator == 'concrete':
                    loss_nll = 0
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for nll in executor.map(concrete_mc, [(X, Z_mu, Z_logstd, logit_alpha, logit_theta, Decoder, args, ii) for ii in range(args.nll_nsample)]):
                            loss_nll += nll/args.nll_nsample
                
                elif args.estimator == 'reinforce':
                    loss_nll = 0
                    eta = args.eta
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for nll, reinforce_loss in executor.map(reinforce_mc, [(X, Z_mu, Z_logstd, logit_alpha, prob_theta, Decoder, args, ii) for ii in range(args.nll_nsample)]):
                            loss_nll += (eta * nll + (1-eta)*old_null  + reinforce_loss - reinforce_loss.detach())/args.nll_nsample                        
                    old_null = loss_nll.detach()

                elif args.estimator == 'soft_approx':
                    sample_Z = reparameterization(Z_mu, Z_logstd) # node latent variables  B X N X H 
                    prob_alpha = torch.softmax(logit_alpha, -1)
                    prob_adj = torch.stack([prob_theta[num] @ prob_alpha[num].reshape(args.num_mix_component, 1) for num in range(args.n_graphs)], dim = 0).squeeze(-1) # B X N X N        
                    Xmu, X_logstd = Decoder(sample_Z.to(device), prob_adj.to(device))
                    loss_nll = nll_gaussian(Xmu, X, X_logstd)

                elif args.estimator == 'measure':
                    loss_nll = 0
                    theta_grads = torch.zeros(X.size(0), args.n_nodes, args.n_nodes, args.num_mix_component).to(device)
                    alpha_grads = torch.zeros(X.size(0), args.num_mix_component).to(device)
                    for _ in range(args.nll_nsample): 
                        sample_Z = reparameterization(Z_mu, Z_logstd).to(device) # node latent variables  B X N X H
                        prob_alpha = torch.softmax(logit_alpha, -1)
                        sample_alpha = torch.multinomial(prob_alpha, num_samples = 1, replacement=True) # B X 1
                        prob_adj = torch.stack([prob_theta[num, :, :, sample_alpha[num]] for num in range(args.n_graphs)], dim = 0).squeeze(-1)
                        sample_adj = torch.bernoulli(prob_adj) # B X N X N
                        Xmu, X_logstd = Decoder(sample_Z, sample_adj.to(device))
                        loss_nll += nll_gaussian(Xmu, X, X_logstd)/args.nll_nsample
                        with torch.no_grad():
                            for kk in range(args.num_mix_component):
                                alpha_k = prob_alpha[:,kk].detach() # 1
                                grad_id = [(ii,jj, kk) for ii in range(args.n_nodes) for jj in range(args.n_nodes)]
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    for thetaijk_grad, idd in executor.map(sample_loss, [(Decoder, idd, sample_Z, prob_theta[:,:,:,kk], alpha_k, X) for idd in grad_id]):
                                        ii, jj, kk = idd
                                        theta_grads[:, ii, jj, kk] += thetaijk_grad                          
                        adj_k = torch.bernoulli(prob_theta[:,:,:,kk]) # B X N X N
                        Xmu, X_logstd = Decoder(sample_Z.to(device), adj_k.to(device))
                        alpha_grads[:, kk] += nll_gaussian(Xmu, X, X_logstd).detach()

                    prob_theta.data -= args.lr * theta_grads/args.nll_nsample
                    prob_alpha.data -= args.lr * alpha_grads/args.nll_nsample
                    theta_grads.zero_()
                    alpha_grads.zero_()

                ## MC sampling for KL divergence
                kl_A, kl_Z = kl_categorical_mc(prob_theta, logit_alpha, args.temp, Z_mu, prior_i, args.klA_nsample) 
                ## Approximate KL divergence
                # kl_A, kl_Z = kl_categorical_approx(prob_theta, logit_alpha, Z_mu, prior_i)
                ## regularized term
                # loss_regu = torch.norm(prob_theta[0], p=1) - torch.sum(torch.diagonal(prob_theta[0], dim1=0, dim2=1, offset=0))
                loss = beta0 * loss_nll + beta1 * kl_A + beta2 * kl_Z 

            scaler.scale(loss).backward()            
            scaler.step(optimizer)
            scaler.update()
            # if epoch < 5:
            #for name, p in model[0].named_parameters():
            #    print(name, torch.norm(p))
            # ema.update()
            epoch_loss += loss
            train_KLAloss += kl_A
            train_KLZloss += kl_Z
            train_REloss += loss_nll   
            prob_alpha = torch.softmax(logit_alpha, -1) # B X K 
    
        scheduler.step()
    
        epoch_loss /= len(train_loader)
        train_KLAloss /= len(train_loader)
        train_KLZloss /= len(train_loader)
        train_REloss /= len(train_loader)
        train_roc_auc, _, _, _, train_ap, *_ = eval(Encoder = Encoder, data_loader=train_loader, device=device) 
        # wandb.log({"prob_theta": wandb.Histogram(prob_theta.detach().cpu().numpy())})     
        # wandb.log({"prob_alpha": wandb.Histogram(prob_alpha.detach().cpu().numpy())})
        # wandb.log({"epoch_loss": epoch_loss, "RE_loss": train_REloss, "A_loss":train_KLAloss, "Z_loss":train_KLZloss, "roc_auc":train_roc_auc, 'ap':train_ap})
        print(f"Epoch {epoch + 1} -- Loss: {epoch_loss.item():.4f} -- A_KLLoss: {train_KLAloss.item():.4f} -- Z_KLLoss: {train_KLZloss:.4f} -- RECONLoss: {train_REloss:.4f} -- ROC_AUC: {train_roc_auc:.4f} -- AP: {train_ap:.4f}. It took {time.time() - start:.2f}s")       
        if epoch == 1:
            t = torch.cuda.get_device_properties(0).total_memory / 1024**2
            r = torch.cuda.memory_reserved(0) / 1024**2
            a = torch.cuda.memory_allocated(0) / 1024**2
            print(f"Total Memory {t}, Reserved Memory {r}, Allocated Memory {a}")
        
        if epoch == 0 or epoch_loss.item() <= min(loss_negELBO):
            torch.save(model.state_dict(), model_path) 

        train_roc_aucs.append(train_roc_auc)
        loss_negELBO.append(epoch_loss.detach().cpu().numpy())

        if epoch > args.patience and min(loss_negELBO[-args.patience:]) > min(loss_negELBO):
            print("Early stopping")
            break
    
    # wandb.finish()
    # torch.save(prob_theta.detach().cpu(), 'gpgvae_theta.pt')
    print("#################################")
    print("Evaluating best model")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    args.model_to_train = 'latent'
              
    ### compute all matrics
    # valid_obj1, valid_obj2 = compute_validloss(Encoder, train_loader, device)
    roc_mean, roc_std, mse_mean, mse_std, ap_mean, ap_std, f1score_mean, f1score_std  = eval(Encoder = Encoder, data_loader=train_loader, device=device)
    print(f"final ROC_AUC: {roc_mean:.4f}+-{roc_std:.4f}")
    print(f"final MSE: {mse_mean:.8f}+-{mse_std:.8f}.")
    print(f"final Averaged Precision: {ap_mean:.4f}+-{ap_std:.4f}.")
    print(f"final F1 score: {f1score_mean:.4f}+-{f1score_std:.4f}.")
    # print(f"final valid_obj1: {valid_obj1:.4f} --obj2:{valid_obj2}.")

    # # baseline methods
    # baseline_results = eval_baseline(new_dataset)
    # models_results["baseline"] = baseline_results


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run(args)
