import numpy as np
import pickle as pkl
from sklearn import covariance
from torch import linalg as LA
from cvxpy import * #  version: 1.1.3; download mosek academic liscence 
from utils import compute_roc_auc_score, compute_average_precision_score, compute_f1_score
import torch
import os
import os.path as osp
import argparse
parser = argparse.ArgumentParser(description='Graphical Lasso')
parser.add_argument('--graph_type', type=str, default='PA_review', help='graph type')
args = parser.parse_args()
## nohup python -u benchmark_algorithms.py PA > benchmark_algorithms_PA.out & 
## nohup python -u benchmark_algorithms.py LA > benchmark_algorithms_LA.out & 


def lasso_baseline_score(X, A_true, alpha):
    # model = GraphicalLasso(alpha = alpha).fit(X.T.cpu().numpy())
    # A_pred = torch.Tensor(model.covariance_)
    n, m = X.shape  
    emp_cov = covariance.empirical_covariance(X.T)
    shrunk_cov = covariance.shrunk_covariance(emp_cov, shrinkage=0.8)# need to tune 
    G, _, costs = covariance.graphical_lasso(shrunk_cov, alpha, return_costs=True)
    obj1 = costs[-1][0]
    A_pred = G * n / G.sum()
    A_pred = torch.Tensor(A_pred)
    A_pred.fill_diagonal_(0.)   
    A_pred[A_pred<0] = 0
    A_pred[A_pred > 1] = 1

    AX = A_pred @ X
    AXT = AX.T
    pinverse = torch.linalg.pinv(AXT@AX)
    obj2 = torch.norm((AX@pinverse@AXT@X - X))/(X.size(0)*X.size(1))
    roc_score = compute_roc_auc_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
    ap_score = compute_average_precision_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
    mmd_mse = LA.matrix_norm(A_true-A_pred)/(A_pred.size(-1)**2)
    f1_score = compute_f1_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
    return obj1, obj2, roc_score, mmd_mse.numpy(), ap_score, f1_score
    

def get_lasso(dataset, alpha):
    roc_aucs = []
    mmd_mses = []
    ap_scores = []
    f1_scores = []
    obj1s = []
    obj2s = []
    for data in dataset:
        A_true = data["A"]
        X = np.log((data["X"] + 0.001))
        obj1, obj2, roc_score, mmd_mse, ap_score, f1_score = lasso_baseline_score(X, A_true, alpha)
        roc_aucs.append(roc_score)
        mmd_mses.append(mmd_mse)
        ap_scores.append(ap_score)
        f1_scores.append(f1_score)
        obj1s.append(obj1)
        obj2s.append(obj2)
    return np.mean(obj1s), np.mean(obj2s), np.mean(roc_aucs), np.std(roc_aucs), np.mean(mmd_mses), np.std(mmd_mses), np.mean(ap_scores), np.std(ap_scores), np.mean(f1_scores), np.std(f1_scores)
          

print(f'graph type: {args.graph_type}')
data_name = os.path.join('Yelp', str(args.graph_type)+'_food.pickle')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
with open(path, 'rb+') as f:
    Data = pkl.load(f)

# Data = pkl.load(open('../data/games/data_barabasi_albert_barik_honorio_-0.6_0_3_100_20.pkl', 'rb+'))
# Graphical Lasso Baseline
print('Tuning Graphical Lasso regularization parameter...', end=' ', flush=True)
alphas = [pow(10, i) for i in range(-5, 2)]   
best_alpha = 0
best_obj = 1e10
objs_ori = []
objs_rec = []
rocs = []
for alpha in alphas:
    obj_ori, obj_rec, train_lasso_rocmean, train_lasso_rocstd, train_lasso_msemean, train_lasso_msestd, train_lasso_apmean, train_lasso_apstd, train_lasso_f1mean, train_lasso_f1std = get_lasso(Data, alpha)
    print(f"Lasso baseline --obj:{obj_ori:.8f} --- ROC_AUC:{train_lasso_rocmean:.4f}+-{train_lasso_rocstd:.4f} --- MSE:{train_lasso_msemean:.4f}+-{train_lasso_msestd:.4f} --- AP:{train_lasso_apmean:.4f}+-{train_lasso_apstd:.4f} --- F1 score:{train_lasso_f1mean:.4f}+-{train_lasso_f1std:.4f}")
    objs_ori.append(obj_ori)
    objs_rec.append(obj_rec)
    rocs.append(train_lasso_rocmean)
#     if obj_new < best_obj:
#         best_obj = obj_new
#         best_alpha = alpha

# print(f'Done! Best glasso regularization parameter found: {best_alpha}')

print('alphas:', alphas)
print('objs_reconstruction:', objs_rec)
print('objs_original:', objs_ori)
print('rocs:', rocs)
print('best alpha:', alphas[np.argmax(rocs)])

# import matplotlib.pyplot as plt
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.plot(alphas, objs_ori)
# plt.xlabel('alpha')
# plt.ylabel('obj_original')
# plt.subplot(1, 3, 2)
# plt.plot(alphas, objs_rec)
# plt.xlabel('alpha')
# plt.ylabel('obj_recon')
# plt.subplot(1, 3, 3)
# plt.plot(alphas, rocs)
# plt.xlabel('alpha')
# plt.ylabel('roc')
# plt.savefig(f'{args.graph_type}_glasso.png')

# _, train_lasso_rocmean, train_lasso_rocstd, train_lasso_msemean, train_lasso_msestd, train_lasso_apmean, train_lasso_apstd, train_lasso_f1mean, train_lasso_f1std = get_lasso(Data, best_alpha)
# print(f"Lasso baseline  --- ROC_AUC:{train_lasso_rocmean:.4f}+-{train_lasso_rocstd:.4f} --- MSE:{train_lasso_msemean:.4f}+-{train_lasso_msestd:.4f} --- AP:{train_lasso_apmean:.4f}+-{train_lasso_apstd:.4f} --- F1 score:{train_lasso_f1mean:.4f}+-{train_lasso_f1std:.4f}")


#### PA review: [10^-5, 10^1], alpha = 0.001,  auc = 0.6989, ap = 0.0394, f1 = 0.0285
#### PA rating: [10^-5, 10^1], alpha = 0.001 ,  auc = 0.7149, ap = 0.0374, f1 = 0.0288
#### LA rating: [10^-5, 10^1], alpha = 0.001, auc = 0.6978, ap=0.0463, f1 = 0.0306
#### LA review: [10^-5, 10^1], alpha = 0.01, auc = 0.7747 ap = 0.0518, f1 = 0.0273

