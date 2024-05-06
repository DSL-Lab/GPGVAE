from cvxpy import Variable, Minimize, Problem, norm, MOSEK, sum_squares, SCS
import mosek


def barik_honorio(X, theta):
    n = X.shape[0]
    G = Variable(shape = (n, n), name = 'G')  
    obj = Minimize(sum_squares(X -  G @ X) +  (theta) * norm(G, 'fro')**2)
    prob = Problem(obj)
    try:
        prob.solve(solver=MOSEK, mosek_params={mosek.iparam.bi_max_iterations:1000}, verbose=False)
    except Exception as e:
        prob.solve(solver=SCS, max_iters = 1000, verbose=False)  
    return G.value

### call function ## 
# Barik_best_param.pkl 
# {'er': 0, 'ws': 0, 'bara': 0}

# call function # 
# barik_honorio(a_com, 10**best_param)