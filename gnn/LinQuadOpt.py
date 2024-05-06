import os.path as osp

import numpy as np
# from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
import copy, math
import pandas as pd # '1.0.1'
from numpy.linalg import norm as nm
import numpy.linalg as LA
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import argparse
# import wandb
import os
import torch
from indian_village_dataset import IndianVillageGames
from games_dataset import Games

import pickle
import math
import warnings
warnings.filterwarnings("ignore")
# torch.set_default_dtype(torch.float64)
 
parser = argparse.ArgumentParser('Games')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--n_graphs', type=int, help='Number of graphs', default=50)
parser.add_argument('--n_nodes', type=int, help='Number of nodes', default=20)
parser.add_argument('--n_games', type=int, help='Number of games', default=100)  
parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=5)
parser.add_argument('--patience', type=int, help='Early Stopping Patience', default=50)
parser.add_argument('--alpha', type=float, help='Smoothness of marginal benefits', default=0)
parser.add_argument('--target_spectral_radius', type=float, help='Target spectral radius', default=-0.6)
parser.add_argument('--theta1', type=float, help='hyparameter', default=1)
parser.add_argument('--theta2', type=float, help='hyparameter', default=1)
parser.add_argument('--graph_type', type=str, help='Type of graph', default="NYC", choices=["barabasi_albert", "erdos_renyi", "watts_strogatz", "indian_village", "PA_review", 'PA_rating', "LA_rating", 'LA_review', "NYC", "TKY", "IST","SaoPaulo","Jakarta","KualaLampur"])
parser.add_argument('--game_type', type=str, help='Type of game', default="realworld", choices=["linear_quadratic", "linear_influence", "barik_honorio", "realworld"])
parser.add_argument('--cost_distribution', type=str, help='Type of distribution to use to sample node-wise costs.', default="normal", choices=["normal", "uniform"])
parser.add_argument('--m', type=int, help='Barabasi-Albert parameter m', default=3)
parser.add_argument('--regenerate_data', action='store_true', help='Whether to regenerate the graphs')
parser.add_argument('--noise_std', type=float, help='B noise std.', default=0.)
parser.add_argument('--action_signal_to_noise_ratio', type=float, help='Signal-to-noise ration in synthetic actions', default=10)


def diagnoal_zero(matrix):
    row = len(matrix)
    col = row
    error = 0.0
    for i in range(row):
        if matrix[i,i] != 0:
            error += matrix[i,i]
    return error

def check_positive(matrix):
    error = 0
    row = len(matrix)
    col = row
    for i in range(row):
        for j in range(col):
            if matrix[i,j] < 0:
                error = error - matrix[i,j]
    return error


def isN(matrix):
    N = len(matrix)
    sum_G=0
    for i in range(N):
        for j in range(N):
            sum_G+=matrix[i,j]
    
    return (sum_G,N,N-sum_G)

def problem_sol(a_com, beta, theta1, theta2, theta3, G, B, alpha): # todo
    n, Dim = a_com.shape
    L = np.zeros((n,n))
    for i in range(n):
        L[i][i] = np.sum(G, 1)[i]
    L = L - G
    L_a = (1 - alpha) * np.identity(n) + alpha * L   ##  ?? why      
    sol = (theta3)* nm((np.identity(n)  - beta * G).dot(a_com)- B, 'fro')**2 + (theta1)* nm(G,'fro')**2 + (theta2) * np.trace(B.T.dot(L_a).dot(B))
    return sol


def isSymmetric(matrix):
    error = 0
    row = len(matrix)
    col = row
    for i in range(row):
        for j in range(col):
            if matrix[i,j] != matrix[j,i]:
                error += pow(matrix[i,j] - matrix[j,i],2) # abs
    return error


def getnew_B(a_com, beta, theta1, theta2, G):
    n, Dim = a_com.shape
    L = np.zeros((n,n))
    for i in range(n):
        L[i][i] = np.sum(G, 1)[i]
    L = L - G
    # L = np.diag(np.sum(G, 1))- G
    B = LA.inv(np.identity(n) + (theta2) * L).dot(np.identity(n) - beta * G).dot(a_com)
    return B


def tune_diagnoal_zero(matrix):
    row = len(matrix)
    col = row
    for i in range(row):
        matrix[i,i] = 0
    return matrix

# Convert continuous predictions to binary based on top N values
def make_binary(predictions, ss):
    threshold = np.quantile(predictions, ss)
    return (predictions > threshold).astype(int)


def compute_f1_score(A_true, A_pred):
    # NOTE: The score is computed only on off-diagonal elements, since we assume the diagonal always contains zero
    sparsity = np.arange(0.1, 1, 0.1)
    
    f1_scores = []
    for ss in sparsity:
        pred = make_binary(A_pred, ss)
        f1_scores.append(f1_score(A_true, pred))

    return np.mean(f1_scores)

def objective(target_beta, theta1, theta2, args, dataset):
    a_com = dataset["X"].numpy()
    A = dataset["A"].numpy()
    # print(a_com.shape, 'shape')

    A_tilde = ( A > 0 ) * 1 # N * N
    N, K = a_com.shape
    ind = np.where(np.tril(np.ones((N, N)), -1).reshape(1,N**2))[1]   # lower triangle index

    beta = - target_beta/max([abs(i) for i in np.linalg.eigvals(A_tilde)])
    alpha = 1e-4 # depends on scale of objective
    
    if args.graph_type in ["PA_review", 'PA_rating', "LA_rating", 'LA_review']:
        emp_cov = np.ones((N,N)) - np.corrcoef(a_com) # negative
    else:
        emp_cov = np.corrcoef(a_com) # positive
    
    tune_diagnoal_zero(emp_cov)
    # print(emp_cov.sum())
    emp_cov = emp_cov * N / emp_cov.sum()          #  L1-norm to N
    G = emp_cov
    A = a_com; Onevec = np.ones((N,1))
    B = np.array(np.random.normal(loc = 0, scale = 1, size = (N, K))) 
    pre_sol = 10000
    for big_k in range(args.n_epochs): 
        BBT = B.dot(B.transpose())

        y_pred = []
        for i in range(len(ind)):
            y_pred.append(G.reshape(N**2,1)[ind[i],0])
        
        y_true = A_tilde.reshape(N**2)[ind].tolist()
        
        roc_score = roc_auc_score(y_true, y_pred)
        ap_score = average_precision_score(y_true, y_pred)
        mmd_mse = nm(np.array(y_true)-np.array(y_pred))/(ind.shape[0])
        f1_sco = compute_f1_score(y_true, y_pred)

        sol = problem_sol(a_com, beta, theta1, theta2, 1, G, B, 1)

        print("Big iteration {}_{}_{}".format(big_k, sol, roc_score))
        # print("isSymmetric, isN, checkpositive diagnoal_zero ", isSymmetric(G), isN(G), check_positive(G), diagnoal_zero(G))
        print('auc', roc_score, 'ap', ap_score, 'f1', f1_sco, 'mse', mmd_mse)
        if abs(pre_sol - sol) <= 0.001:
            break 
       
        for k in range(50):        
            # compute gradient for G
            pre_G = copy.deepcopy(G)
            GA = pre_G.dot(A)
            gradient_1 = 2 * beta * (beta * GA + B - A).dot(A.transpose())
            gradient_2 = 2 * theta1 * G
            gradient_3 = theta2 * np.diagonal(BBT).reshape(N,1).dot(Onevec.transpose())  - theta2 * BBT
            deltaG = gradient_1 + gradient_2 + gradient_3
            deltaG = (deltaG + deltaG.T)/2
            
            pre_G = pre_G.reshape(N**2)
            G = G.reshape(N**2)
            deltaG = deltaG.reshape(N**2)
            denominator = 0.0

            for i in range(N**2):
                try:
                    denominator += pre_G[i] * math.exp(-1 * alpha * deltaG[i])
                except OverflowError:
                    denominator += pre_G[i] * math.exp(600)
            
            for i in range(N**2):
                try:
                    G[i] = (N * pre_G[i] * math.exp(-1 * alpha * deltaG[i]))/denominator
                except OverflowError:
                    G[i] = (N * pre_G[i] * math.exp(600))/denominator
            
            G = G.reshape(N,N)
            # if k % 10 == 0:
            #     y_pred = []
            #     for i in range(len(ind)):
            #         y_pred.append(G.reshape(N**2,1)[ind[i],0])
                        
            #     df = pd.DataFrame({'y_true': A_tilde.reshape(N**2)[ind].tolist(), 'y_pred':y_pred}).sort_values('y_pred')[::-1]
            #     roc = roc_auc_score(df.y_true.values, df.y_pred.values)
            #     auc = round(roc, 3)    
            #     print("Small iteration {}_{}_{}".format(k, sol, auc))
        B = getnew_B(a_com, beta, theta1, theta2, G)

    # pickle.dump(G, open('linquadopt_inferred_graph.pkl', 'wb'))
    # with open('G.pkl', 'wb') as f:
    #     pickle.dump(G, f)

    y_pred = []
    for i in range(len(ind)):
        y_pred.append(G.reshape(N**2,1)[ind[i],0])
        
    y_true = A_tilde.reshape(N**2)[ind].tolist()
        
    roc_score = roc_auc_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    mmd_mse = nm(np.array(y_true)-np.array(y_pred))/(ind.shape[0])
    f1_sco = compute_f1_score(y_true, y_pred)
    obj_ori = (LA.norm((np.identity(N) -  beta * G) @ a_com - B, ord='fro')**2 +  (theta1) * LA.norm(G, ord='fro')**2 + (theta2) * LA.norm(B, 2) **2 )
    G = torch.tensor(G)
    G.fill_diagonal_(0.)   
    G[G<0] = 0
    G[G > 1] = 1
    X = torch.tensor(a_com, dtype=torch.float64)
    AX = G @ X
    AXT = AX.T
    pinverse = torch.linalg.pinv(AXT@AX)
    obj_rec = torch.norm((AX@pinverse@AXT@X - X))
    
    print("Parameter is " + str(target_beta) + ' ' + str(theta1) + ' ' + str(theta2) +  " auc is " + str(roc_score), 
        'ap', ap_score, 'f1', f1_sco, 'mse', mmd_mse, 'obj_ori', obj_ori, 'obj_rec', obj_rec)
    
    return obj_ori, obj_rec, roc_score, ap_score, f1_sco, mmd_mse



### nyc: beta = 0.9, theta1 = 1e-10, theta2 = 1e-7
### sp: beta = 0.7, theta1 = 1e-8, theta2 = 1e-8
### village: beta = 0.1, theta1 = 1e-10, theta2 = 1e-10
### PA rating: beta = 0.9, theta1 =1e-7 , theta2 =1e-7
### PA review: beta = 0.9, theta1 = 1e-10 , theta2 = 1e-10
### LA rating: beta = 0.9  , theta1 =1e-15, theta2 =1e-15 
### LA review: beta = 0.9  , theta1 =1e-7, theta2 =1e-7

if __name__ == "__main__":
    args = parser.parse_args()
    args.n_epochs = 5
    print(args)
    # seed_everything(args.seed)
    if args.graph_type == 'indian_village':
        data_name = 'indian_networks'      
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        dataset = IndianVillageGames(path)
        new_dataset = []
        for i in range(len(dataset)):
            A = torch.FloatTensor(dataset[i]['A'])
            X = dataset[i]['X']
            X = torch.FloatTensor(X)
            new_dataset.append({"X": X, "A": A})

    elif args.graph_type == 'PA_rating' or args.graph_type == 'PA_review' or args.graph_type == 'LA_rating' or args.graph_type == 'LA_review':       
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
            print(f'number of uses: {X.shape[0]}, number of businesses: {X.shape[1]}')
            new_dataset.append({"X": X, "A": A})  
        print(f'number of graphs: {len(new_dataset)}')
        
    elif args.graph_type == 'NYC' or args.graph_type == 'IST' or args.graph_type == 'TKY' or args.graph_type == 'SaoPaulo' or args.graph_type == 'Jakarta' or args.graph_type == 'KualaLampur':
        data_name = os.path.join('Foursquare', str(args.graph_type)+'_dataset.pickle')
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        with open(path, 'rb+') as f:
            dataset = pickle.load(f)   
        new_dataset = []   
        for i in range(len(dataset)):
            X = torch.log(dataset[i]['X'] + 1e-3)
            A = torch.FloatTensor(dataset[i]['A'])  
            X = torch.FloatTensor(X)       
            new_dataset.append({"X": X, "A": A})                

    elif args.graph_type == 'barabasi_albert' or args.graph_type == 'erdos_renyi' or args.graph_type == 'watts_strogatz':
        data_name = 'games'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_name)
        new_dataset = Games(path, n_graphs=args.n_graphs, n_games=args.n_games, n_nodes=args.n_nodes, m=args.m, 
                    target_spectral_radius=args.target_spectral_radius, alpha=args.alpha,
                    signal_to_noise_ratio=args.action_signal_to_noise_ratio, game_type=args.game_type,
                    regenerate_data=args.regenerate_data, graph_type=args.graph_type,  cost_distribution=args.cost_distribution)
    else:
        raise NotImplementedError   
    
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # theta1s = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    # theta2s = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    objs_ori = []
    objs_rec = []
    rocs_beta = []
    for beta in betas:
        args.target_spectral_radius = beta
        aucs = []
        f1s = []
        aps = []
        mses = []
        objs = []
        for i in range(len(new_dataset)):
            data = new_dataset[i]
            obj1, obj2, roc_auc, ap_score, f1_sco, mmd_mse = objective(args.target_spectral_radius, args.theta1, args.theta2, args, data)
            aucs.append(roc_auc)
            f1s.append(f1_sco)
            aps.append(ap_score)
            mses.append(mmd_mse)

        print(f"Average ROC_AUC: {np.mean(aucs):.4f}+-{np.std(aucs):.4f}")
        # print(f"Average Averaged Precision: {np.mean(aps):.4f}+-{np.std(aps):.4f}.")
        # print(f'Average F1 score: {np.mean(f1s):.4f}+-{np.std(f1s):.4f}.')
        # print(f"Average MSE: {np.mean(mses):.8f}+-{np.std(mses):.8f}.")
        # print(f"Average obj: {np.mean(objs):.8f}+-{np.std(objs):.8f}.")

        objs_ori.append(obj1)
        objs_rec.append(obj2)
        rocs_beta.append(np.mean(aucs))

    print('graph type: ', args.graph_type)
    print('beta: ', betas)
    print('obj_ori: ', objs_ori)
    print('obj_rec: ', objs_rec)
    print('roc_auc: ', rocs_beta)
    print('best beta: ', betas[np.argmax(rocs_beta)])


    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.plot(betas, objs_ori)
    # plt.xlabel('beta')
    # plt.ylabel('obj_original')
    # plt.subplot(1, 3, 2)
    # plt.plot(betas, objs_rec)
    # plt.xlabel('beta')
    # plt.ylabel('obj_recon')
    # plt.subplot(1, 3, 3)
    # plt.plot(betas, rocs_beta)
    # plt.xlabel('beta')
    # plt.ylabel('roc_auc')
    # plt.savefig(f'{args.graph_type}_linquadopt_beta.png')
    # plt.show()
