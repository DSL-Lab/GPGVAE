import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import linalg as LA
import networkx as nx
from utils import *


def get_correlation(dataset, anticorrelation):
    roc_aucs = []
    mmd_mses = []
    ap_scores = []
    f1_scores = []
    # graph_ref_list = []
    # graph_pred_list = [] 
    for data in dataset:
        A_true = data["A"]
        roc_score, mmd_mse, ap_socre, f1_score = correlation_baseline_score(data["X"], A_true, anticorrelation=anticorrelation)
        roc_aucs.append(roc_score)
        mmd_mses.append(mmd_mse)
        ap_scores.append(ap_socre)
        f1_scores.append(f1_score)
        # graph_ref_list.append(nx.from_numpy_matrix(np.asmatrix(A_true)))
        # graph_pred_list.append(nx.from_numpy_matrix(np.asmatrix(tA_pred)))
    # mmd_degree = degree_stats(graph_ref_list, graph_pred_list)
    # mmd_clustering = clustering_stats(graph_ref_list, graph_pred_list)   
    # mmd_spectral = spectral_stats(graph_ref_list, graph_pred_list)  
    return np.mean(roc_aucs), np.std(roc_aucs), np.mean(mmd_mses), np.std(mmd_mses), np.mean(ap_scores), np.std(ap_scores), np.mean(f1_scores), np.std(f1_scores)


def get_barik_honorio_roc_auc(dataset, theta, train):
    if train == True:
        objs = [barik_honorio_score(data["X"], theta) for data in dataset]
        return np.mean(objs)
    else:  
        roc_aucs = []
        mmd_mses = []
        ap_scores = []
        f1_scores = []
        # graph_ref_list = []
        # graph_pred_list = [] 
        for data in dataset:
            X, A_true = data["X"], data["A"]
            pred = barik_honorio(X.cpu().numpy(), theta)
            A_pred = torch.Tensor(pred)
            A_pred.fill_diagonal_(0.)   
            A_pred[A_pred<0] = 0
            A_pred[A_pred > 1] = 1
            roc_score = compute_roc_auc_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
            ap_score = compute_average_precision_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
            mmd_mse = LA.matrix_norm(A_true-A_pred)/(A_pred.size(-1)**2)
            f1_score = compute_f1_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
            # tA_pred = matrix_top_processing(A_pred, sparsity)
            roc_aucs.append(roc_score)
            mmd_mses.append(mmd_mse)
            ap_scores.append(ap_score)
            f1_scores.append(f1_score)
        #     graph_ref_list.append(nx.from_numpy_matrix(np.asmatrix(A_true)))
        #     graph_pred_list.append(nx.from_numpy_matrix(np.asmatrix(tA_pred)))
        # mmd_degree = degree_stats(graph_ref_list, graph_pred_list)
        # mmd_clustering = clustering_stats(graph_ref_list, graph_pred_list)   
        # mmd_spectral = spectral_stats(graph_ref_list, graph_pred_list)  
        return np.mean(roc_aucs), np.std(roc_aucs), np.mean(mmd_mses), np.std(mmd_mses), np.mean(ap_scores), np.std(ap_scores), np.mean(f1_scores), np.std(f1_scores)
           
        
def get_linear_quadratic_optimization_roc_auc(dataset, beta, theta1, theta2, smooth, train):
    if train == True:
        objs = [linear_quadratic_optimization_score(data["X"], beta, theta1, theta2, smooth) for data in dataset]
        return np.mean(objs)
    else:   
        roc_aucs = []
        mmd_mses = []
        ap_scores = []
        f1_scores = []
        # graph_ref_list = []
        # graph_pred_list = [] 
        for data in dataset:
            X, A_true = data["X"], data["A"]
            # pred, _ = non_smoothing_model(X.cpu().numpy(), beta, theta1, theta2)
            pred, _ = smoothing_model(X.cpu().numpy(), beta, theta1, theta2)
            A_pred = torch.Tensor(pred)
            A_pred.fill_diagonal_(0.)   
            A_pred[A_pred<0] = 0
            A_pred[A_pred > 1] = 1
            roc_score = compute_roc_auc_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
            ap_score = compute_average_precision_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
            # tA_pred = matrix_top_processing(A_pred, sparsity)
            mmd_mse = LA.matrix_norm(A_true-A_pred)/(A_pred.size(-1)**2)
            f1_score = compute_f1_score(A_true.unsqueeze(0), A_pred.unsqueeze(0))
            roc_aucs.append(roc_score)
            mmd_mses.append(mmd_mse)
            ap_scores.append(ap_score)
            f1_scores.append(f1_score)
        #     graph_ref_list.append(nx.from_numpy_matrix(np.asmatrix(A_true)))
        #     graph_pred_list.append(nx.from_numpy_matrix(np.asmatrix(tA_pred)))
        # mmd_degree = degree_stats(graph_ref_list, graph_pred_list)
        # mmd_clustering = clustering_stats(graph_ref_list, graph_pred_list)   
        # mmd_spectral = spectral_stats(graph_ref_list, graph_pred_list)  
        return np.mean(roc_aucs), np.std(roc_aucs), np.mean(mmd_mses), np.std(mmd_mses), np.mean(ap_scores), np.std(ap_scores), np.mean(f1_scores), np.std(f1_scores)
        

def get_lasso(dataset, alpha, train = True):
    if train == True:
        objs = []
        for data in dataset:
            obj = lasso_baseline_score(data["X"], data["A"], alpha, train)
            objs.append(obj)
        return np.mean(objs)
    else:
        roc_aucs = []
        mmd_mses = []
        ap_scores = []
        f1_scores = []
        # graph_ref_list = []
        # graph_pred_list = [] 
        for data in dataset:
            A_true = data["A"]
            roc_score, mmd_mse, ap_score, f1_score = lasso_baseline_score(data["X"], A_true, alpha, train = False)
            roc_aucs.append(roc_score)
            mmd_mses.append(mmd_mse)
            ap_scores.append(ap_score)
            f1_scores.append(f1_score)
        #     graph_ref_list.append(nx.from_numpy_matrix(np.asmatrix(A_true)))
        #     graph_pred_list.append(nx.from_numpy_matrix(np.asmatrix(tA_pred)))
        # mmd_degree = degree_stats(graph_ref_list, graph_pred_list)
        # mmd_clustering = clustering_stats(graph_ref_list, graph_pred_list)   
        # mmd_spectral = spectral_stats(graph_ref_list, graph_pred_list)  
        return np.mean(roc_aucs), np.std(roc_aucs), np.mean(mmd_mses), np.std(mmd_mses), np.mean(ap_scores), np.std(ap_scores), np.mean(f1_scores), np.std(f1_scores)
       

def eval_baseline(train_dataset):
    # Correlation Baseline
    train_corr_rocmean, train_corr_rocstd, train_corr_msemean, train_corr_msestd, train_corr_apmean, train_corr_apstd, train_corr_f1mean, train_corr_f1std = get_correlation(train_dataset, anticorrelation=0)
    print(f"Correlation baseline --- ROC_AUC:{train_corr_rocmean:.4f}+-{train_corr_rocstd:.4f} --- MSE:{train_corr_msemean:.4f}+-{train_corr_msestd:.4f} --- AP:{train_corr_apmean:.4f}+-{train_corr_apstd:.4f} --- F1 score:{train_corr_f1mean:.4f}+-{train_corr_f1std:.4f}")

    # Anticorrelation Baseline
    train_anticorr_rocmean, train_anticorr_rocstd, train_anticorr_msemean, train_anticorr_msestd, train_anticorr_apmean, train_anticorr_apstd, train_anticorr_f1mean, train_anticorr_f1std  = get_correlation(train_dataset, anticorrelation=1)
    print(f"Anticorrelation baseline --- ROC_AUC:{train_anticorr_rocmean:.4f}+-{train_anticorr_rocstd:.4f} --- MSE:{train_anticorr_msemean:.4f}+-{train_anticorr_msestd:.4f} --- AP:{train_anticorr_apmean:.4f}+-{train_anticorr_apstd:.4f} --- F1 score:{train_anticorr_f1mean:.4f}+-{train_anticorr_f1std:.4f}")

    # Graphical Lasso Baseline
    print('Tuning Graphical Lasso regularization parameter...', end=' ', flush=True)
    alphas = [pow(2, (2*i-10)) for i in range(5)]
    best_alpha = 0
    best_obj = 1e4
    for alpha in alphas:
         obj_new = get_lasso(train_dataset, alpha, train = True)
         if obj_new < best_obj:
             best_obj = obj_new
             best_alpha = alpha
    print(f'Done! Best glasso regularization parameter found: {best_alpha}')
    train_lasso_rocmean, train_lasso_rocstd, train_lasso_msemean, train_lasso_msestd, train_lasso_apmean, train_lasso_apstd, train_lasso_f1mean, train_lasso_f1std = get_lasso(train_dataset, best_alpha, train = False)
    print(f"Lasso baseline  --- ROC_AUC:{train_lasso_rocmean:.4f}+-{train_lasso_rocstd:.4f} --- MSE:{train_lasso_msemean:.4f}+-{train_lasso_msestd:.4f} --- AP:{train_lasso_apmean:.4f}+-{train_lasso_apstd:.4f} --- F1 score:{train_lasso_f1mean:.4f}+-{train_lasso_f1std:.4f}")

    # barik_honorio
    best_theta = 0
    thetas = [pow(2, (2*i-5)) for i in range(6)]
    best_obj = 1e4
    for theta in thetas:
         obj_new = get_barik_honorio_roc_auc(train_dataset, theta, train = True)
         if obj_new < best_obj:
             best_obj = obj_new
             best_theta = theta
    print(f'Done! Best BH parameter found: {best_theta}')
    train_bh_rocmean, train_bh_rocstd, train_bh_msemean, train_bh_msestd, train_bh_apmean, train_bh_apstd, train_bh_f1mean, train_bh_f1std = get_barik_honorio_roc_auc(train_dataset, best_theta, train = False)

    print(f"BH baseline  --- ROC_AUC:{train_bh_rocmean:.4f}+-{train_bh_rocstd:.4f} --- MSE:{train_bh_msemean:.4f}+-{train_bh_msestd:.4f} --- AP:{train_bh_apmean:.4f}+-{train_bh_apstd:.4f}, --- F1 score:{train_bh_f1mean:.4f}+-{train_bh_f1std:.4f}")

    # # LinearQuadratic smooth = true
    # betas = [-0.6]
    # best_sfbeta = 0
    # best_sftheta1 = 0
    # best_sftheta2 = 0
    # theta1s = [pow(2, (2*i-3)) for i in range(1)]
    # theta2s = [pow(2, (2*i-3)) for i in range(1)]
    # best_obj = 1e4
    # for beta in betas:
    #     for theta1 in theta1s:
    #         for theta2 in theta2s:
                # obj_new = get_linear_quadratic_optimization_roc_auc(train_dataset, beta, theta1, theta2, smooth = False, train = True)
    #             if obj_new < best_obj:
    #                 best_obj = obj_new
    #                 best_sfbeta = beta
    #                 best_sftheta1 = theta1
    #                 best_sftheta2 = theta2
    # print(f'Done! Best smooth parameter found: {[best_sfbeta, best_sftheta1, best_sftheta2 ]}')
    # train_sfquadratic_rocmean, train_sfquadratic_rocstd, train_sfquadratic_msemean, train_sfquadratic_msestd, train_sfquadratic_apmean, train_sfquadratic_apstd, train_sfquadratic_f1mean, train_sfquadratic_f1std = get_linear_quadratic_optimization_roc_auc(train_dataset, best_sfbeta, best_sftheta1, best_sftheta2, smooth = False, train = False)
    # print(f"Quadratic smooth baseline  --- ROC_AUC:{train_sfquadratic_rocmean:.4f}+-{train_sfquadratic_rocstd:.4f} --- MSE:{train_sfquadratic_msemean:.4f}+-{train_sfquadratic_msestd:.4f} --- AP:{train_sfquadratic_apmean:.4f}+-{train_sfquadratic_apstd:.4f}, --- F1 score:{train_sfquadratic_f1mean:.4f}+-{train_sfquadratic_f1std:.4f}")


    baseline_results = {
            "correlation_train_roc_auc_mean": train_corr_rocmean, "correlation_train_roc_auc_std": train_corr_rocstd,
            "anticorrelation_train_roc_auc_mean": train_anticorr_rocmean, "anticorrelation_train_roc_auc_std": train_anticorr_rocstd,
            "lasso_train_roc_auc_mean": train_lasso_rocmean, "lasso_train_roc_auc_std": train_lasso_rocstd,
            # "quadratic_train_roc_auc_mean": train_sfquadratic_rocmean, "quadratic_train_roc_auc_std": train_sfquadratic_rocstd,
            "BH_train_roc_auc_mean": train_bh_rocmean, "BH_train_roc_auc_std": train_bh_rocstd,
            "correlation_train_mse_mean": train_corr_msemean, "correlation_train_mse_std": train_corr_msestd,
            "anticorrelation_train_mse_mean": train_anticorr_msemean, "anticorrelation_train_mse_std": train_anticorr_msestd,
            "lasso_train_mse_mean": train_lasso_msemean, "lasso_train_mse_std": train_lasso_msestd,
            # "quadratic_train_mse_mean": train_sfquadratic_msemean, "quadratic_train_mse_std": train_sfquadratic_msestd,
            "BH_train_mse_mean": train_bh_msemean, "BH_train_mse_std": train_bh_msestd,
            "correlation_train_ap_mean": train_corr_apmean, 'correlation_train_ap_std': train_corr_apstd,
            "anticorrelation_train_ap_mean": train_anticorr_apmean, 'anticorrelation_train_ap_std': train_anticorr_apstd,
            "lasso_train_ap_mean": train_lasso_apmean, 'lasso_train_ap_std': train_lasso_apstd,
            # "quadratic_train_ap_mean": train_sfquadratic_apmean, 'quadratic_train_ap_std': train_sfquadratic_apstd,
            "BH_train_ap_mean": train_bh_apmean, 'BH_train_ap_std': train_bh_apstd,
            "correlation_train_f1_mean": train_corr_f1mean, 'correlation_train_f1_std': train_corr_f1std,
            "anticorrelation_train_f1_mean": train_anticorr_f1mean, 'anticorrelation_train_f1_std': train_anticorr_f1std,
            "lasso_train_f1_mean": train_lasso_f1mean, 'lasso_train_f1_std': train_lasso_f1std,
            # "quadratic_train_f1_mean": train_sfquadratic_f1mean, 'quadratic_train_f1_std': train_sfquadratic_f1std,
            "BH_train_f1_mean": train_bh_f1mean, 'BH_train_f1_std': train_bh_f1std
            }
    return baseline_results

def sample_binary(prob_theta):
    sample_A = torch.bernoulli(prob_theta)
    return sample_A

def eval(Encoder, data_loader, device):
    Encoder.eval()
    roc_aucs = []
    mmd_mses = []
    ap_scores = []
    f1_scores = []
    objs = []
    with torch.no_grad():
        for data in data_loader:
            X, A_true, length, _ = data
            X, A_true, length = X.to(device), A_true.to(device), length.to(device)
            n_graphs = X.size(0)
            # prob_theta, logit_alpha, *_= model(X) 
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

                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     for sample_prob in executor.map(sample_binary, [prob_theta[j, :, :, aa] for aa in alpha_list]):
                #         A_pred_j += sample_prob/n_samples            
               
                # idx = torch.argmax(prob_alpha[j])
                # A_pred_j = prob_theta[j, :, :, idx]

                roc_score = compute_roc_auc_score(A_true_j, A_pred_j)
                ap_score = compute_average_precision_score(A_true_j, A_pred_j)
                roc_aucs.append(roc_score) 
                ap_scores.append(ap_score)
                mmd_mse = torch.sum(LA.matrix_norm(A_true_j-A_pred_j)/(A_pred_j.size(-1)**2))
                mmd_mses.append(mmd_mse.cpu().detach().numpy())
                f1_score = compute_f1_score(A_true_j, A_pred_j)
                f1_scores.append(f1_score)

        return np.mean(roc_aucs), np.std(roc_aucs), np.mean(mmd_mses), np.std(mmd_mses), np.mean(ap_scores), np.std(ap_scores), np.mean(f1_scores), np.std(f1_scores)
   
def compute_validloss(Encoder, data_loader, device):
    Encoder.eval()
    valid_obj1s = []
    valid_obj2s = []
    for data in data_loader:
        valid_obj1 = 0
        valid_obj2 = 0
        X, A_true, length, _ = data
        n_graphs, n, m = X.shape
        X, A_true, length = X.to(device), A_true.to(device), length.to(device)
        logit_theta, logit_alpha, *_ = Encoder(X)
        prob_theta = torch.stack([mask_diagonal_and_sigmoid(logit_theta[:,:,:,kk]) for kk in range(logit_theta.size(-1))], dim = -1) 
        prob_alpha = F.softmax(logit_alpha, -1)
        n_samples = 5000
        for j in range(n_graphs):
            alpha_list = torch.multinomial(prob_alpha[j], n_samples, replacement=True).tolist()
            A_true_j = A_true[j:j+1, :length[j], :length[j]]
                
            A_pred_j = torch.zeros_like(A_true_j)
            for ii in range(n_samples):
                A_pred_j += sample_binary(prob_theta[j, :, :, alpha_list[ii]].detach())/n_samples

            A_pred_j = A_pred_j[0]
            A_pred_j.fill_diagonal_(0.)
            A_pred_j = A_pred_j / (A_pred_j.max() + 1e-8)

            ## l1 norm and l2 norm
            diags = A_pred_j.sum(axis=1)  
            diags = 1.0 / torch.sqrt(diags)
            D = torch.diag_embed(diags)  
            AX = D @ (A_pred_j @ D) @ X[j]  
            ### l1 norm
            W = torch.nn.parameter.Parameter(torch.randn((m, m), requires_grad=True, device=device))
            lr = 1e-4
            for epoch in range(500):
                l1_loss = torch.norm(AX@W - X, p=1)/n
                l1_loss.backward()
                W.data -= lr * W.grad
                W.grad.zero_()
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch [{epoch+1}], Loss: {l1_loss.item()}')
    
            obj1 = l1_loss.item()
            ## l2 norm 
            AXT = AX.T
            pinverse = torch.linalg.pinv(AXT@AX)
            obj2 = torch.norm((AX@pinverse@AXT@X - X))/n
            # W2 = torch.nn.parameter.Parameter(torch.randn(K, K), requires_grad=True)
            # lr = 1e-3
            # for epoch in range(100):
            #     l2_loss = torch.norm(AX@W2 - X, p='fro')/(N)
            #     l2_loss.backward()
            #     W2.data -= lr * W2.grad
            #     W2.grad.zero_()
            #     if (epoch + 1) % 1 == 0:
            #         print(f'Epoch {epoch+1}, l2 Loss: {l2_loss.item()}')
            # obj2 = l2_loss.item()
            valid_obj1 += obj1
            valid_obj2 += obj2
        valid_obj1s.append(valid_obj1)
        valid_obj2s.append(valid_obj2.item())

    return np.mean(valid_obj1s), np.mean(valid_obj2s)
