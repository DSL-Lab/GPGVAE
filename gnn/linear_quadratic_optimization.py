import random 
import mosek
import numpy as np 
from numpy.linalg import norm as nm
from numpy.linalg import inv, pinv
import cvxpy as cp
from cvxpy import Variable, trace, Minimize, sum, Problem, norm, MOSEK, ECOS, diag, sum_squares, SCS, pnorm, vec, CVXOPT

seed = 1
def smoothing_model(a_com, beta, theta1, theta2, max_iter = 50, tol = 1e-5):
    # a_com = a_com[:200,:200]
    n, Dim = a_com.shape
    B = np.random.RandomState(seed=seed).multivariate_normal(mean = np.zeros(n), cov = np.identity(n), size = Dim).T
    cnt = 0
    delta_prob_value = 1
    pre_sol = 1000000 
    while delta_prob_value >= tol:
        if cnt >= max_iter:
            return G.value
        cnt += 1
        G = Variable(shape = (n, n), name = 'G', symmetric=True)  
        L = diag(sum(G, 1))- G
        # obj = Minimize(sum_squares((np.identity(n)  -  beta * G) @ a_com - B) +  (theta1) * pnorm(vec(G), 2)**2  + (theta2) * trace(B.T @ L @ B))
        obj = Minimize(sum_squares((np.identity(n)  -  beta * G) @ a_com - B) +  (theta1) * sum_squares(vec(G))  + (theta2) * trace(B.T @ L @ B))
        constraint = [diag(G) == 0]
        prob = Problem(obj, constraint); 
        try:
            prob.solve(solver=MOSEK, mosek_params={mosek.iparam.bi_max_iterations:1000}, verbose=True)
        except Exception as e:
            prob.solve(solver=SCS, max_iters = 1000, verbose=False) 
        L = np.diag(np.sum(G.value, 1))- G.value
        B = inv(np.identity(n) + (theta2) * L).dot(np.identity(n) - beta *  G.value).dot(a_com)
        sol = nm((np.identity(n)  - beta * G.value).dot(a_com)- B, 'fro')**2 + (theta1)* nm( G.value,'fro')**2 + (theta2) * np.trace(B.T.dot(L).dot(B))
        delta_prob_value = abs(sol - pre_sol)
        pre_sol = sol
    return G.value

def non_smoothing_model(a_com, beta, theta1, theta2):
    n, Dim = a_com.shape
    
    G = Variable((n,n), symmetric = True)  
    B = Variable(shape = (n, Dim), name = 'B')
    obj = Minimize(sum_squares((np.identity(n) -  beta * G) @ a_com - B) +  (theta1) * pnorm(vec(G), 2) **2 + (theta2) * pnorm(vec(B), 2) **2 )
    constraint = [diag(G) == 0, G >=0, sum(G) == n]
    prob = Problem(obj, constraint)
    try:
        # prob.solve(solver = CVXOPT)
        prob.solve(solver=MOSEK, mosek_params={mosek.iparam.bi_max_iterations:1000}, verbose=False)
    except Exception as e:
        prob.solve(solver=SCS, max_iters = 1000, verbose=False)
    return G.value
