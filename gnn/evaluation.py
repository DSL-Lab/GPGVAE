import numpy as np
import torch
import torch.nn.functional as F
from utils import mask_diagonal_and_sigmoid, compute_roc_auc_score


def sample_binary(prob_theta):
    sample_A = torch.bernoulli(prob_theta)
    return sample_A


def eval(Encoder, data_loader, device):
    Encoder.eval()
    roc_aucs = []
    with torch.no_grad():
        for data in data_loader:
            X, A_true, length, _ = data
            X, A_true, length = X.to(device), A_true.to(device), length.to(device)
            n_graphs = X.size(0)
            logit_theta, logit_alpha, *_ = Encoder(X)
            prob_theta = torch.stack([mask_diagonal_and_sigmoid(logit_theta[:,:,:,kk]) for kk in range(logit_theta.size(-1))], dim = -1) 
            prob_alpha = F.softmax(logit_alpha, -1)
            n_samples = 5000
            K = prob_theta.size(-1)
            for j in range(n_graphs):
                A_true_j = A_true[j:j+1, :length[j], :length[j]]
                if K == 1:
                    A_pred_j = prob_theta[j, :, :, 0]
                else:
                    alpha_list = torch.multinomial(prob_alpha[j], n_samples, replacement=True).tolist()
                    A_pred_j = torch.zeros_like(A_true_j)
                    for ii in range(n_samples):
                        A_pred_j += sample_binary(prob_theta[j, :, :, alpha_list[ii]])/n_samples
                roc_score = compute_roc_auc_score(A_true_j, A_pred_j)   
                roc_aucs.append(roc_score)       

        return np.mean(roc_aucs), np.std(roc_aucs)
    