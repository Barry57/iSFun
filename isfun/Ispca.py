import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

def ispca(x, L, mu1, mu2, eps = 1e-04, pen1 = "homogeneity", 
          pen2 = "magnitude", scale_x = True, maxstep = 50, 
          submaxstep = 10, trace = False, draw = False):
    if not isinstance(x, list):
        raise TypeError("x should be of list type.")
    x = [np.array(i) for i in x] 
    nl = [i.shape[0] for i in x]
    pl = [i.shape[1] for i in x] 
    p = list(set(pl))
    if len(p) > 1:
        raise ValueError("The dimension of data x should be consistent among different datasets.")
    if len(p) == 1:
        p = p[0]
    ip = np.array(range(1, p+1))
    meanx = [np.dot(np.ones((1, nl[l])), x[l])/nl[l] for l in range(L)]
    for l in range(L):
        x[l] = scale(x[l], with_std=False)
    def x_scale(l):
        one = np.ones((1, nl[l]))
        normx = np.sqrt(np.dot(one, x[l]**2) / (nl[l] - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some of the columns of the predictor matrix have zero variance.")
        return normx
    if scale_x:
        normx = [x_scale(i) for i in range(L)]
    else:
        normx = [[1]*p[0]]*L   
    if scale_x:
        x = [(x[i] - np.mean(x[i], axis=0)) / normx[i] for i in range(L)]
    def ro(x, mu, alpha):
        def f(x):
            return mu * np.where(x/(mu * alpha) < 1, 1 - x/(mu * alpha), 0)
        if np.isscalar(x):
            r = quad(f, 0, x)[0]
        else:
            r = np.array([quad(f, 0, xi)[0] for xi in x])
        return r
    def ro_d1st(x, mu, alpha):
        r = mu * (1 > x/(mu * alpha)) * (1 - x/(mu * alpha))
        return r
    def u_value_homo(u, v, p, L, mu1, mu2, pen2):
        def fun_s(j, l):
            s1 = np.sum(x[l] * np.tile(v[l], (p)).reshape(nl[l], p), axis=0) / nl[l]
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(u[j, np.arange(len(u[0]))!=l])
            if pen2 == "sign":
                s2 = mu2 * np.sum([x / np.sqrt(x**2 + 0.5) for x in u[j, np.arange(len(u[0]))!=l]]) / np.sqrt(u[j, l]**2 + 0.5)
            s = s1[j] + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)])
        s = result_s.reshape(p, L)
        norm_u_j = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, u)
        ro_d = ro_d1st(norm_u_j, mu1, 6)
        def fun_c(j, l):
            s_norm = np.sqrt(np.sum(s[j, :]**2))
            if pen2 == "magnitude":
                c = nl[l] * (s_norm > ro_d[j]) * s[j, l] * (s_norm - ro_d[j]) / ((1 + mu2 * nl[l] * (L - 1)) * s_norm)
            if pen2 == "sign":
                c = nl[l] * (s_norm > ro_d[j]) * s[j, l] * (s_norm - ro_d[j]) / ((1 + mu2 * nl[l] * (L - 1) / (u[j, l]**2 + 0.5)) * s_norm)
            return c
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    def u_value_hetero(u, v, p, L, mu1, mu2, pen2):
        def fun_mu(j, l):
            u_j = u[j, :]
            ro_j = ro(np.abs(u_j), mu1, 6)
            s_ro = np.sum(ro_j)
            mu_jl = ro_d1st(s_ro, 1, 1/2 * L * 6 * mu1**2) * ro_d1st(abs(u_j[l]), mu1, 6)
            return mu_jl
        result_mu = np.array([fun_mu(j, l) for j in range(p) for l in range(L)])
        mu = result_mu.reshape(p, L)
        def fun_s(j, l):
            s1 = np.sum(x[l] * np.tile(v[l], (p)).reshape(nl[l], p), axis=0) / nl[l]
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(u[j, np.arange(len(u[0]))!=l])
            if pen2 == "sign":
                s2 = mu2 * np.sum([x / np.sqrt(x**2 + 0.5) for x in u[j, np.arange(len(u[0]))!=l]]) / np.sqrt(u[j, l]**2 + 0.5)
            s = s1[j] + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)])
        s = result_s.reshape(p, L)
        def fun_c(j, l):
            if pen2 == "magnitude":
                c = nl[l] * np.sign(s[j, l]) * (abs(s[j, l]) > mu[j, l]) * (abs(s[j, l]) - mu[j, l]) / (1 + mu2 * nl[l] * (L - 1))
            if pen2 == "sign":
                c = nl[l] * np.sign(s[j, l]) * (abs(s[j, l]) > mu[j, l]) * (abs(s[j, l]) - mu[j, l]) / (1 + mu2 * nl[l] * (L - 1) / (u[j, l]**2 + 0.5))
            return c   
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    what = np.zeros((p, L))
    def fun_1(l):
        u_l, s_l, v_l = svds(x[l], k=1)
        return v_l.T * s_l[0]
    U = np.hstack(np.array([fun_1(l) for l in range(L)]))
    def fun_2(l):
        u_l, s_l, v_l = svds(x[l], k=1)
        return u_l
    V = [fun_2(l) for l in range(L)]
    sgn = [np.sign(np.dot(U[:, 0], U[:, l]) / (np.sqrt(np.sum(U[:, 0]**2)) * np.sqrt(np.sum(U[:, l]**2)))) for l in range(1, L)]
    for l in range(1, L):
        U[:, l] = sgn[l - 1] * U[:, l]
        V[l] = sgn[l - 1] * V[l]
    u = U.copy()
    v = V.copy()
    iter = 1
    dis_u_iter = 10
    loading_trace = np.zeros((p * L, maxstep))
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis_u_iter > eps and iter <= maxstep:
        u_iter = u.copy()
        subiter_u = 1
        dis_u = 10
        while dis_u > eps and subiter_u <= submaxstep:
            u_old = u.copy()
            if pen1 == "homogeneity":
                u = u_value_homo(u, v, p, L, mu1, mu2, pen2)
            if pen1 == "heterogeneity":
                u = u_value_hetero(u, v, p, L, mu1, mu2, pen2)
            u_norm = np.sqrt(np.sum(u**2, axis=0))
            u_norm[u_norm == 0] = 1e-04
            u_scale = (u.T / np.expand_dims(u_norm, axis=-1)).T  
            dis_u = np.max(np.sqrt(np.sum((u - u_old)**2, axis=0)) / np.sqrt(np.sum(u_old**2, axis=0)))
            what = u_scale
            what_cut = np.where(np.abs(what) > 1e-04, what, 0)
            what_cut_norm = np.sqrt(np.sum(what_cut**2, axis=0))
            what_cut_norm[what_cut_norm == 0] = 1e-04
            what_dir = (what_cut.T / np.expand_dims(what_cut_norm, axis=-1)).T
            what_cut = np.where(np.abs(what_dir) > 1e-04, what_dir, 0)
            loading_trace[:, iter-1] = what_cut.ravel(order='F')
            if np.sum(np.sum(np.abs(u_scale), axis=0) <= 1e-04) == p:
                break
            subiter_u += 1
        if np.sum(np.sum(np.abs(u), axis=0) <= 1e-04) == p:
            print("Stop! The value of mu1 is too large")
            what_cut = u
            break  
        v = [np.expand_dims(x[l].dot(u[:, l]) / np.sqrt(np.sum((x[l].dot(u[:, l])**2))), axis=-1) for l in range(L)]
        dis_u_iter = np.max(np.sqrt(np.sum((u - u_iter)**2, axis=0)) / np.sqrt(np.sum(u_iter**2, axis=0)))
        if np.sum(np.sum(np.abs(u), axis=0) <= 1e-04) == p:
            break
        if trace:
            new2A = [np.where(what_cut[:, l] != 0)[0] for l in range(L)]
            print("\n--------------------\n----- Step {} -----\n--------------------".format(iter))
            for l in range(L):
                new2A_l = new2A[l]
                if len(new2A_l) <= 10:
                    print("DataSet {}:".format(l+1))
                    print(', '.join(['X{}'.format(i+1) for i in new2A_l]))
                else:
                    print("DataSet {}:".format(l+1))
                    nlines = int(np.ceil(len(new2A_l) / 10))
                    for i in range(nlines - 1):
                        print(', '.join(['X{}'.format(i+1) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                    print(', '.join(['X{}'.format(i+1) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
        iter += 1
    loading_trace = loading_trace[:, :iter-1]
    what = what_cut
    new2A = [ip[what[:, l] != 0] for l in range(L)] 
    if draw:
        for l in range(L):
            plt.figure()
            plt.plot(loading_trace[l * p: (l + 1) * p, :].T)
            plt.title(f"Dataset {l+1}\nConvergence path of elements in vector u")
            plt.xlabel("Number of iterations")
            plt.ylabel("Weight")
            plt.show()
    eigenvalue = [what[:, l].T.dot(np.cov(x[l], rowvar = False)).dot(what[:, l]) for l in range(L)]
    comp = [x[l].dot(what[:, l]) for l in range(L)]
    dictname = [f'Dataset {l+1}' for l in range(L)]
    new2A_dict = {}
    meanx_dict = {}
    normx_dict = {}
    x_dict = {}
    eigenvalue_dict = {}
    comp_dict = {}
    for l in range(L):
        new2A_dict.update({dictname[l]:new2A[l]})
        meanx_dict.update({dictname[l]:meanx[l]})
        normx_dict.update({dictname[l]:normx[l]})      
        x_dict.update({dictname[l]:x[l]})
        eigenvalue_dict.update({dictname[l]:eigenvalue[l]})
        comp_dict.update({dictname[l]:comp[l]})
    what_df = pd.DataFrame(what, index = [str(i+1) for i in range(p)], columns = dictname)
    class ispca:
        def __init__(self, data):
            self.x = data['x']
            self.eigenvalue = data['eigenvalue']
            self.eigenvector = data['eigenvector']
            self.component = data['component']
            self.variable = data['variable']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.pen1 = data['pen1']
            self.pen2 = data['pen2']
            self.mu1 = data['mu1']
            self.mu2 = data['mu2']
            self.loading_trace = data['loading_trace']
    object_dict = {
        'x': x_dict, 
        'eigenvalue': eigenvalue_dict, 
        'eigenvector': what_df, 
        'component': comp_dict, 
        'variable': new2A_dict, 
        'meanx': meanx_dict,
        'normx': normx_dict, 
        'pen1': pen1, 
        'pen2': pen2, 
        'mu1': mu1, 
        'mu2': mu2, 
        'loading_trace': loading_trace
    }
    object = ispca(object_dict)
    return object