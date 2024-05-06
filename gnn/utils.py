import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from encoder import MLPEncoder, TransformerEncoder
from decoder import GCNDecoder


def compute_roc_auc_score(A_true, A_pred):
    # NOTE: The ROC_AUC score is computed only on off-diagonal elements, since we assume the diagonal always contains zero
    pred = mask_diagonal(A_pred).reshape(-1).cpu().detach().numpy()
    target = mask_diagonal(A_true).reshape(-1).cpu().detach().numpy()

    return roc_auc_score(target, pred)


def mask_diagonal(A):
    if len(A.shape) == 2:
        mask = ~torch.eye(A.shape[1], dtype=bool, device=A.device)
    elif len(A.shape) == 3:
        mask = ~torch.eye(A.shape[1], dtype=bool, device=A.device).repeat(A.shape[0],1,1)
    return A[mask]
    #return A.masked_select(mask)


def mask_diagonal_and_sigmoid(A):
    mask = ~torch.eye(A.shape[1], dtype=bool, device=A.device).repeat(A.shape[0],1,1)
    return torch.sigmoid(A)*mask


def get_encoder(encoder_type, n_games, n_heads, hidden_dim, num_mix_component):
    if encoder_type == "mlp_on_seq":
        return MLPEncoder(n_games, hidden_dim)
    elif encoder_type == "transformer":
        return TransformerEncoder(hidden_dim, n_heads, n_games, num_mix_component)
    else:
        return NotImplementedError

def get_decoder(hidden_dim, out_channels, n_layer):
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


def kl_categorical_mc(prob_theta, logit_alpha, temp, Z_mu, prior, N_samples): 
    n_graphs, n_nodes = Z_mu.size(0), Z_mu.size(1) 
    K = prob_theta.size(-1)   # number of mixture components  
    logit_theta = logitstic(prob_theta).reshape(n_graphs, n_nodes*n_nodes, K)  # B X N X N X K

    loss_KLA = 0
    for _ in range(N_samples):
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



##  Source code for torch_geometric.utils._to_dense_adj
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