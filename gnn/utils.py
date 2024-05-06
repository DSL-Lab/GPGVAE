import numpy as np
import scipy.linalg
import networkx as nx
import concurrent.futures
import torch
import torch.nn.functional as F
from sklearn import covariance

from sklearn.metrics import roc_auc_score, average_precision_score
from torch.autograd import Variable
from sklearn import metrics
from torch.autograd import Function
from scipy.linalg import eigvalsh
from functools import partial
from torch import linalg as LA
from encoder import MLPEncoder, PerGameTransformerEncoder, GCN_ZEncoder, TransformerEncoder, MLPEncoder2, TransformerZEncoder
from decoder import GCNDecoder, GatedGCNDecoder
from barik_honorio_model import barik_honorio
from linear_quadratic_optimization import non_smoothing_model, smoothing_model
from sklearn.covariance import GraphicalLasso


def compute_roc_auc_score(A_true, A_pred):
    # NOTE: The ROC_AUC score is computed only on off-diagonal elements, since we assume the diagonal always contains zero
    pred = mask_diagonal(A_pred).reshape(-1).cpu().detach().numpy()
    target = mask_diagonal(A_true).reshape(-1).cpu().detach().numpy()

    return roc_auc_score(target, pred)

def compute_average_precision_score(A_true, A_pred):
    ## only on off-diagonal elements
    pred = mask_diagonal(A_pred).reshape(-1).cpu().detach().numpy()
    target = mask_diagonal(A_true).reshape(-1).cpu().detach().numpy()

    return average_precision_score(target, pred)

def compute_recall_score(A_true, A_pred):
    # NOTE: The score is computed only on off-diagonal elements, since we assume the diagonal always contains zero
    pred = mask_diagonal(A_pred).reshape(-1).cpu().detach().numpy()
    target = mask_diagonal(A_true).reshape(-1).cpu().detach().numpy()

    return metrics.recall_score(target, pred)

def compute_precision_score(A_true, A_pred):
    # NOTE: The score is computed only on off-diagonal elements, since we assume the diagonal always contains zero
    pred = mask_diagonal(A_pred).reshape(-1).cpu().detach().numpy()
    target = mask_diagonal(A_true).reshape(-1).cpu().detach().numpy()

    return metrics.precision_score(target, pred)

def compute_f1_score(A_true, A_pred):
    # NOTE: The score is computed only on off-diagonal elements, since we assume the diagonal always contains zero
    A_pred = mask_diagonal(A_pred).reshape(-1).cpu().detach().numpy()
    target = mask_diagonal(A_true).reshape(-1).cpu().detach().numpy()
    sparsity = np.arange(0.1, 1, 0.1)
    
    f1_scores = []
    for ss in sparsity:
        pred = A_pred.copy()
        threshold = np.quantile(pred, ss)
        pred[A_pred > threshold] = 1
        pred[A_pred <= threshold] = 0
        f1_scores.append(metrics.f1_score(target, pred))

    return np.mean(f1_scores)

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    ''' Discrepancy between 2 samples '''
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for dist in executor.map(kernel_parallel_worker, [
    #       (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
    #   ]):
    #     d += dist

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker, [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]):
                d += dist
    d /= len(samples1) * len(samples2)
    return d

def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d

def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)

def gaussian_tv(x, y, sigma=1.0):  
    support_size = max(len(x), len(y))
      # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))

def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' MMD between two samples '''
      # normalize histograms into pmf  
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
           disc(samples2, samples2, kernel, *args, **kwargs) - \
           2 * disc(samples1, samples2, kernel, *args, **kwargs)

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
          graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
  # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    # prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
              nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
      # elapsed = datetime.now() - prev
      # if PRINT_TIME:
      #   print('Time computing degree mmd: ', elapsed)
    return mmd_dist

def spectral_worker(G):
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())  
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)
        
        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            print(clustering_coeffs_list)
            hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    return mmd_dist

def mask_diagonal(A):
    if len(A.shape) == 2:
        mask = ~torch.eye(A.shape[1], dtype=bool, device=A.device)
    elif len(A.shape) == 3:
        mask = ~torch.eye(A.shape[1], dtype=bool, device=A.device).repeat(A.shape[0],1,1)
    return A[mask]
    #return A.masked_select(mask)

def mask_diagonal_and_sigmoid2(A):
    mask = torch.eye(A.shape[1], dtype=bool, device=A.device).repeat(A.shape[0],1,1)
    return torch.sigmoid(A - mask*100000000)

def mask_diagonal_and_sigmoid(A):
    mask = ~torch.eye(A.shape[1], dtype=bool, device=A.device).repeat(A.shape[0],1,1)
    return torch.sigmoid(A)*mask

def zero_out_diagonal(A):
    for i in range(A.shape[0]):
        A[i].fill_diagonal_(0.)

def correlation_baseline_score(x, A_true, anticorrelation):
    multiplier = -1 if anticorrelation==1 else 1
    corr_coeff = torch.corrcoef(torch.Tensor(x))
    corr_coeff = (corr_coeff + 1) / 2
    A_pred = multiplier * corr_coeff
    A_pred.fill_diagonal_(0.)   
    roc_score = compute_roc_auc_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
    ap_score = compute_average_precision_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
    mmd_mse = LA.matrix_norm(A_true-A_pred)/(A_pred.size(-1)**2)
    f1_score = compute_f1_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
    # tA_pred = matrix_top_processing(A_pred, sparsity)    

    return roc_score,  mmd_mse.numpy(), ap_score, f1_score

def lasso_baseline_score(X, A_true, alpha, train):
    # model = GraphicalLasso(alpha = alpha).fit(X.T.cpu().numpy())
    # A_pred = torch.Tensor(model.covariance_)
    n, m = X.shape  
    emp_cov = covariance.empirical_covariance(X.T)
    shrunk_cov = covariance.shrunk_covariance(emp_cov, shrinkage=alpha)# need to tune 
    alpha = 0.1 # need to tune 
    G, _ = covariance.graphical_lasso(shrunk_cov, alpha)
    A_pred = G * n / G.sum()
    A_pred = torch.Tensor(A_pred)
    A_pred.fill_diagonal_(0.)   
    A_pred[A_pred<0] = 0
    A_pred[A_pred > 1] = 1

    if train == True:
        AX = A_pred @ X
        AXT = AX.T
        pinverse = torch.linalg.pinv(AXT@AX)
        obj = torch.norm((AX@pinverse@AXT@X - X))
        return obj
    else: 
        roc_score = compute_roc_auc_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
        ap_score = compute_average_precision_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
        # tA_pred = matrix_top_processing(A_pred, sparsity)
        mmd_mse = LA.matrix_norm(A_true-A_pred)/(A_pred.size(-1)**2)
        f1_score = compute_f1_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
        return roc_score, mmd_mse.numpy(), ap_score, f1_score
    

def barik_honorio_score(X, theta):
    pred = barik_honorio(X.cpu().numpy(), theta)
    A_pred = torch.Tensor(pred)
    A_pred[A_pred<0] = 0
    A_pred[A_pred > 1] = 1
    A_pred.fill_diagonal_(0.) 
    AX = A_pred @ X
    AXT = AX.T
    pinverse = torch.linalg.pinv(AXT@AX)
    obj = torch.norm((AX@pinverse@AXT@X - X))
    return obj

def linear_quadratic_optimization_score(X, beta, theta1, theta2, smooth):
    if smooth:
        A_pred = smoothing_model(X.cpu().numpy(), beta, theta1, theta2)[0]
        A_pred = torch.Tensor(A_pred)
    else:
        A_pred = non_smoothing_model(X.cpu().numpy(), beta, theta1, theta2)[0]
        A_pred =  torch.Tensor(A_pred)
    A_pred.fill_diagonal_(0.)   
    A_pred[A_pred<0] = 0
    A_pred[A_pred > 1] = 1
    AX = A_pred @ X
    AXT = AX.T
    pinverse = torch.linalg.pinv(AXT@AX)
    obj = torch.norm((AX@pinverse@AXT@X - X))
    return obj


def permute_features(X, B):
    p = np.random.permutation(X.shape[-1])
    X = X[:, :, p]
    B = B[:, :, p]

    return X, B

def normalize_adjacency(adj_matrix):
    # Calculate the sum of each row in the adjacency matrix
    row_sum = torch.sum(np.abs(adj_matrix), dim=1)
    diags_sqrt = 1.0 / (np.sqrt(row_sum+1e-8))
    diags_sqrt[np.isinf(diags_sqrt)] = 0.0
    
    # Create a diagonal matrix with the reciprocal of the row sums
    d_inverse_sqrt = torch.diag(diags_sqrt)
    
    # Normalize the adjacency matrix using the diagonal matrix
    normalized_adj_matrix = d_inverse_sqrt @ adj_matrix @ d_inverse_sqrt
    
    return normalized_adj_matrix

def get_encoder(encoder_type, n_games, n_heads, hidden_dim, n_nodes, num_mix_component):
    if encoder_type == "mlp_on_seq":
        return MLPEncoder(n_games, hidden_dim)
    elif encoder_type == "gcn_on_z":
        return GCN_ZEncoder(n_games, hidden_dim, n_nodes)
    elif encoder_type == "transformer_z":
        return TransformerZEncoder(hidden_dim, n_heads, n_games)
    elif encoder_type == "per_game_transformer":
        return PerGameTransformerEncoder(hidden_dim, n_heads, num_mix_component)
    elif encoder_type == "mlp_encoder_2":
        return MLPEncoder2(hidden_dim, n_heads, n_games, num_mix_component)
    elif encoder_type == "transformer":
        return TransformerEncoder(hidden_dim, n_heads, n_games, num_mix_component)
    else:
        return NotImplementedError

def get_decoder(hidden_dim, out_channels, n_layer):
    # return GatedGCNDecoder(hidden_dim, out_channels, n_layer)
    return GCNDecoder(hidden_dim, out_channels, n_layer)

def reparameterization(mu, logstd):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu: n_graph*n_nodes*hidden_dim
        :param var:
        :return: sampled z
        """
        sigma = torch.exp(logstd)
        eps = torch.randn_like(mu)
        return mu + sigma * eps  

def matrix_top_processing(A, sparsity):
    n_nodes = A.shape[0]  
    flatten_A = torch.zeros_like(A.reshape(1, -1))   # 1 X N*N
    _, indices = torch.sort(A.reshape(1,-1), descending=True)   # 1 X N*N
    flatten_A[:, indices[0, :int((1-sparsity)*n_nodes*n_nodes)]] = 1.0
    flatten_A[:, indices[0, int((1-sparsity)*n_nodes*n_nodes):]] = 0.0
    top_A = flatten_A.reshape(n_nodes, n_nodes)
    return top_A


def kl_categorical_prior(prob_theta, Z_mu, prior): 
    n_nodes = prob_theta.size(1)
    adj_loss = adj_loss_func(prob_theta, prior).reshape(-1, prob_theta.size(1)*prob_theta.size(1))  # B X N^2
    adj_loss = torch.sum(adj_loss, dim = -1)/n_nodes  # B X 1
    loss_KLA = torch.mean(adj_loss)
    loss_KLZ = 0.5*torch.pow(Z_mu, 2)
    loss_KLZ = torch.sum(loss_KLZ)/(Z_mu.size(0)*n_nodes)

    return loss_KLA, loss_KLZ


def kl_categorical_approx(prob_theta, logit_alpha, Z_mu, prior): 
    n_graphs, n_nodes = Z_mu.size(0), Z_mu.size(1) 
    K = prob_theta.size(-1)   # number of mixture components  
      
    adj_loss = torch.stack([adj_loss_func(prob_theta[:,:,:,kk], prior) for kk in range(K)], dim=1)  # B X K X N X N
    adj_loss = adj_loss.reshape(n_graphs, K, n_nodes*n_nodes)
    adj_loss = torch.sum(adj_loss, dim = -1)  # B X K

    log_alpha = F.log_softmax(logit_alpha, -1)  # B X K
    log_prob = adj_loss + log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1) # B X 1
    loss_KLA = torch.mean(log_prob)/(n_nodes * n_nodes) 

    loss_KLZ = 0.5*torch.pow(Z_mu, 2)
    loss_KLZ = torch.mean(loss_KLZ)

    return loss_KLA, loss_KLZ


def kl_categorical_mc(prob_theta, logit_alpha, temp, Z_mu, prior, n_sample): 
    n_graphs, n_nodes = Z_mu.size(0), Z_mu.size(1) 
    K = prob_theta.size(-1)   # number of mixture components  
    logit_theta = logitstic(prob_theta).reshape(n_graphs, n_nodes*n_nodes, K)  # B X N X N X K

    # NOTE Muchen: MCMC eating up memory N_samples 100 -> 40
    N_samples = n_sample
    loss_KLA = 0
    for _ in range(N_samples):
        # print(_, torch.cuda.memory_allocated() /(1024*1024))
        sample_alpha = gumbel_softmax(logit_alpha, tau=temp, hard=True) # B X K  use hard sample trick to get one discrete category for mixture components
        logit_adj = torch.stack([torch.mm(logit_theta[num], sample_alpha[num].reshape(K, 1)) for num in range(n_graphs)], dim = 0)
        logit_adj = logit_adj.reshape(n_graphs, n_nodes, n_nodes)
        sample_A = binary_concrete(logit_adj, tau=temp, hard=True)  # B X N X N, use hard sample trick to get discrete sampling       
        adj_loss = torch.stack([adj_loss_mc(sample_A, prob_theta[:,:,:,kk], prior) for kk in range(K)], dim=1)  # B X K X N X N
        adj_loss = adj_loss.reshape(n_graphs, K, n_nodes*n_nodes)
        adj_loss = torch.sum(adj_loss, dim = -1) # B X K

        log_alpha = F.log_softmax(logit_alpha, -1)  # B X K
        log_prob = adj_loss + log_alpha
        log_prob = torch.logsumexp(log_prob, dim=1) # B X 1
        loss_sampleA = torch.mean(log_prob)/(n_nodes * n_nodes) 
        loss_KLA += loss_sampleA / N_samples

    loss_KLZ = 0.5*torch.pow(Z_mu, 2)
    loss_KLZ = torch.mean(loss_KLZ)
    
    return loss_KLA, loss_KLZ


def log_Normal_diag(x, mean):
    # log_normal = -0.5 * (torch.log(torch.tensor(2*np.pi))+log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    log_normal = -0.5* torch.pow( x - mean, 2)
    return torch.mean(log_normal, dim = -1) 
    
def adj_loss_func(A_prob, prior, eps = 1e-5):  
    loss_sub = A_prob * (torch.log(A_prob + eps) - torch.log(prior + eps)) + (1-A_prob) * (torch.log((1-A_prob) + eps) - torch.log((1 - prior) + eps))
    return loss_sub

def adj_loss_mc(A_sample, A_prob, prior_sets, eps = 1e-5): 
    loss_sub = (A_sample * (torch.log(A_prob + eps) - torch.log(prior_sets + eps)) + (1-A_sample) * (torch.log((1-A_prob) + eps) - torch.log((1 - prior_sets) + eps)))
    return loss_sub


def nll_gaussian(X_mu, X, X_logstd, add_const=False):
    variance = torch.exp(2*X_logstd)
    neg_log_p = X_logstd + torch.div(torch.pow(X_mu - X, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi))
        neg_log_p += const
    return torch.sum(neg_log_p)/(X.size(0)*X.size(1)*X.size(2))

    
def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def my_sigmoid(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = torch.sigmoid(trans_input)
    return soft_max_1d.transpose(axis, 0)

def logitstic(x, eps = 1e-5):
    return torch.log(x + eps) - torch.log(1 - x + eps)

def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)

def binary_concrete_sample(logits1, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits1.size(), eps=eps)
    if logits1.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits1 + Variable(logistic_noise)  # n_nodes * n_nodes
    return y

def binary_concrete(logits1, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits1, tau=tau, eps=eps)
    y_soft = 1 / (1 + torch.exp(- y_soft / tau))
    
    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        # y = Variable(y_hard.data - y_soft.data) + y_soft
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim = -1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
       Given a positive semi-definite matrix X,
       X = X^{1/2}X^{1/2}, compute the gradient: dX^{1/2} by solving the Sylvester equation, 
       dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
    """
    @staticmethod
    def forward(ctx, input):
        #m = input.numpy().astype(np.float_)
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real)#.type_as(input)
        ctx.save_for_backward(sqrtm) # save in cpu
        sqrtm = sqrtm.type_as(input)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            #sqrtm, = ctx.saved_variables
            sqrtm, = ctx.saved_tensors
            #sqrtm = sqrtm.data.numpy().astype(np.float_)
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            #gm = grad_output.data.numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
#            gm = np.eye(grad_output.shape[-1])
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return Variable(grad_input)
    


from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


def to_dense_adj(
    edge_index: Tensor,
    batch: OptTensor = None,
    edge_attr: OptTensor = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
        ...                            [0, 1, 0, 3, 0]])
        >>> batch = torch.tensor([0, 0, 1, 1])
        >>> to_dense_adj(edge_index, batch)
        tensor([[[1., 1.],
                [1., 0.]],
                [[0., 1.],
                [1., 0.]]])

        >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
        tensor([[[1., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]])

        >>> edge_attr = torch.Tensor([1, 2, 3, 4, 5])
        >>> to_dense_adj(edge_index, batch, edge_attr)
        tensor([[[1., 2.],
                [3., 0.]],
                [[0., 4.],
                [5., 0.]]])
    """
    if batch is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = edge_index.new_zeros(num_nodes)

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    elif ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
          or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
    adj = adj.view(size)

    return adj