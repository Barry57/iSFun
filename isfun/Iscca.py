import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

def iscca(x, y, L, mu1, mu2, mu3, mu4, eps=1e-04, pen1="homogeneity", 
          pen2="magnitude", scale_x=True, scale_y=True, maxstep=50, 
          submaxstep=10, trace=False, draw=False):
    if not isinstance(x, list):
        raise TypeError("x should be of list type.")
    if not isinstance(y, list):
        raise TypeError("y should be of list type.")
    x = [np.array(i) for i in x]
    y = [np.array(i) for i in y]  
    nl = [i.shape[0] for i in x]
    pl = [i.shape[1] for i in x]    
    ql = [i.shape[1] for i in y]  
    p = list(set(pl))
    q = list(set(ql))   
    if len(p) > 1:
        raise ValueError("The dimension of data x should be consistent among different datasets.")
    if len(q) > 1:
        raise ValueError("The dimension of data y should be consistent among different datasets.")
    if len(p) == 1:
        p = p[0]
    if len(q) == 1:
        q = q[0] 
    ip = np.array(range(1, p+1))
    iq = np.array(range(1, q+1))
    meanx = [np.dot(np.ones((1, nl[l])), x[l])/nl[l] for l in range(L)]
    meany = [np.dot(np.ones((1, nl[l])), y[l])/nl[l] for l in range(L)]
    for l in range(L):
        x[l] = scale(x[l], with_std=False)
        y[l] = scale(y[l], with_std=False)
    def x_scale(l):
        one = np.ones((1, nl[l]))
        normx = np.sqrt(np.dot(one, x[l]**2) / (nl[l] - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some of the columns of the predictor matrix have zero variance.")
        return normx
    def y_scale(l):
        one = np.ones((1, nl[l]))
        normy = np.sqrt(np.dot(one, y[l]**2) / (nl[l] - 1))
        if np.any(normy < np.finfo(float).eps):
            raise ValueError("Some of the columns of the response matrix have zero variance.")
        return normy
    if scale_x:
        normx = [x_scale(i) for i in range(L)]
    else:
        normx = [[1]*p[0]]*L   
    if scale_y:
        normy = [y_scale(i) for i in range(L)]
    else:
        normy = [[1]*q[0]]*L
    if scale_x:
        x = [(x[i] - np.mean(x[i], axis=0)) / normx[i] for i in range(L)]
    if scale_y:
        y = [(y[i] - np.mean(y[i], axis=0)) / normy[i] for i in range(L)]    
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
    def u_value_homo(u, v, p, q, L, mu1, mu2, pen2):
        def fun_s(j, l):
            s1 = np.dot(v[:, l].T, y[l].T).dot(x[l][:, j].reshape(1, -1).T) / nl[l]
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(u[j, np.arange(len(u[0]))!=l])
            if pen2 == "sign":
                s2 = mu2 * np.sum([x / np.sqrt(x**2 + 0.5) for x in u[j, np.arange(len(u[0]))!=l]]) / np.sqrt(u[j, l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)])
        s = result_s.reshape(p, L)
        norm_u_j = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, u)
        ro_d = ro_d1st(norm_u_j, mu1, 6)
        def fun_c(j, l):
            s_norm = np.sqrt(np.sum(s[j, :]**2))
            if pen2 == "magnitude":
                c = (s_norm > ro_d[j]) * s[j, l] * (s_norm - ro_d[j]) / ((mu2 * (L - 1)) * s_norm)
            if pen2 == "sign":
                c = (s_norm > ro_d[j]) * s[j, l] * (s_norm - ro_d[j]) / ((mu2 * (L - 1) / (u[j, l]**2 + 0.5)) * s_norm)
            return c
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    def u_value_hetero(u, v, p, q, L, mu1, mu2, pen2):
        def fun_mu(j, l):
            u_j = u[j, :]
            ro_j = ro(np.abs(u_j), mu1, 6)
            s_ro = np.sum(ro_j)
            mu_jl = ro_d1st(s_ro, 1, 1/2 * L * 6 * mu1**2) * ro_d1st(abs(u_j[l]), mu1, 6)
            return mu_jl
        result_mu = np.array([fun_mu(j, l) for j in range(p) for l in range(L)])
        mu = result_mu.reshape(p, L)
        def fun_s(j, l):
            s1 = np.dot(v[:, l].T, y[l].T).dot(x[l][:, j].reshape(1, -1).T) / nl[l]
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(u[j, np.arange(len(u[0]))!=l])
            if pen2 == "sign":
                s2 = mu2 * np.sum([x / np.sqrt(x**2 + 0.5) for x in u[j, np.arange(len(u[0]))!=l]]) / np.sqrt(u[j, l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)])
        s = result_s.reshape(p, L)
        def fun_c(j, l):
            if pen2 == "magnitude":
                c = np.sign(s[j, l]) * (abs(s[j, l]) > mu[j, l]) * (abs(s[j, l]) - mu[j, l]) / (mu2 * (L - 1))
            if pen2 == "sign":
                c = np.sign(s[j, l]) * (abs(s[j, l]) > mu[j, l]) * (abs(s[j, l]) - mu[j, l]) / (mu2 * (L - 1) / (u[j, l]**2 + 0.5))
            return c
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    def v_value_homo(u, v, p, q, L, mu1, mu2, pen2):
        def fun_s(j, l):
            s1 = np.dot(u[:, l].T, x[l].T).dot(y[l][:, j].reshape(1, -1).T) / nl[l]
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(v[j, np.arange(L)!=l])
            if pen2 == "sign":
                s2 = mu2 * np.sum([x/np.sqrt(x**2 + 0.5) for x in v[j, np.arange(L)!=l]]) / np.sqrt(v[j, l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = [fun_s(j, l) for j in range(q) for l in range(L)]
        s = np.array(result_s).reshape(q, L)
        norm_v_j = np.sqrt(np.sum(v**2, axis=1))
        ro_d = ro_d1st(norm_v_j, mu1, 6)
        def fun_c(j, l):
            s_norm = np.sqrt(np.sum(s[j, :]**2))
            if pen2 == "magnitude":
                c = (s_norm > ro_d[j]) * s[j, l] * (s_norm - ro_d[j]) / ((mu2 * (L - 1)) * s_norm)
            if pen2 == "sign":
                c = (s_norm > ro_d[j]) * s[j, l] * (s_norm - ro_d[j]) / ((mu2 * (L - 1) / (v[j, l]**2 + 0.5)) * s_norm)
            return c
        c = np.array([fun_c(j, l) for j in range(q) for l in range(L)]).reshape(q, L)
        return c
    def v_value_hetero(u, v, p, q, L, mu1, mu2, pen2):
        def fun_mu(j, l):
            v_j = v[j, :]
            ro_j = ro(np.abs(v_j), mu1, 6)
            s_ro = np.sum(ro_j)
            mu_jl = ro_d1st(s_ro, 1, 1/2 * L * 6 * mu1**2) * ro_d1st(abs(v_j[l]), mu1, 6)
            return mu_jl
        result_mu = [fun_mu(j, l) for j in range(q) for l in range(L)]
        mu = np.array(result_mu).reshape(q, L)
        def fun_s(j, l):
            s1 = np.dot(u[:, l].T, x[l].T).dot(y[l][:, j].reshape(1, -1).T) / nl[l]
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(v[j, np.arange(L)!=l])
            if pen2 == "sign":
                s2 = mu2 * np.sum([x/np.sqrt(x**2 + 0.5) for x in v[j, np.arange(L)!=l]]) / np.sqrt(v[j, l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = [fun_s(j, l) for j in range(q) for l in range(L)]
        s = np.array(result_s).reshape(q, L)
        def fun_c(j, l):
            if pen2 == "magnitude":
                c = np.sign(s[j, l]) * (abs(s[j, l]) > mu[j, l]) * (abs(s[j, l]) - mu[j, l]) / (mu2 * (L - 1))
            if pen2 == "sign":
                c = np.sign(s[j, l]) * (abs(s[j, l]) > mu[j, l]) * (abs(s[j, l]) - mu[j, l]) / (mu2 * (L - 1) / (u[j, l]**2 + 0.5))
            return c
        c = np.array([fun_c(j, l) for j in range(q) for l in range(L)]).reshape(q, L)
        return c
    what = np.zeros((p, L))
    def fun_1(l):
        Z_l = svds(np.dot(x[l].T, y[l]), k=1)
        u_l = Z_l[0]
        return u_l
    U = np.hstack(np.array([fun_1(l) for l in range(L)]))
    def fun_2(l):
        Z_l = svds(np.dot(x[l].T, y[l]), k=1)
        v_l = Z_l[2]
        return v_l
    V = np.vstack(np.array([fun_2(l) for l in range(L)])).T
    sgn1 = [np.sign(np.dot(np.dot(U[:, l], x[l].T), np.dot(y[l], V[:, l]))) for l in range(L)]
    for l in range(L):
        V[:, l] = sgn1[l] * V[:, l]
    sgn2 = [np.sign(np.dot(U[:, 0], U[:, l]) / (np.sqrt(np.sum(U[:, 0]**2)) * np.sqrt(np.sum(U[:, l]**2)))) for l in range(1, L)]
    for l in range(1, L):
        U[:, l] = sgn2[l - 1] * U[:, l]
        V[:, l] = sgn2[l - 1] * V[:, l]
    u = U.copy()
    v = V.copy()
    iter = 1
    dis_u_iter = 10
    dis_v_iter = 10
    loading_trace_u = np.zeros((p * L, maxstep))
    loading_trace_v = np.zeros((q * L, maxstep))
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis_u_iter > eps and dis_v_iter > eps and iter <= maxstep:
        u_iter = u.copy()
        v_iter = v.copy()
        subiter_u = 1
        dis_u = 10
        while dis_u > eps and subiter_u <= submaxstep:
            u_old = u.copy()
            if pen1 == "homogeneity":
                u = u_value_homo(u, v, p, q, L, mu1, mu2, pen2)
            if pen1 == "heterogeneity":
                u = u_value_hetero(u, v, p, q, L, mu1, mu2, pen2)
            u_norm = np.sqrt(np.sum(u**2, axis=0))
            u_norm[u_norm == 0] = 1e-04
            u = (u.T / np.expand_dims(u_norm, axis=-1)).T
            dis_u = np.max(np.sqrt(np.sum((u - u_old)**2, axis=0)) / np.sqrt(np.sum(u_old**2, axis=0)))
            what = u
            what_cut = np.where(np.abs(what) > 1e-04, what, 0)
            what_cut_norm = np.sqrt(np.sum(what_cut**2, axis=0))
            what_cut_norm[what_cut_norm == 0] = 1e-04
            what_dir = (what_cut.T / np.expand_dims(what_cut_norm, axis=-1)).T
            what_cut = np.where(np.abs(what_dir) > 1e-04, what_dir, 0)
            if np.sum(np.sum(np.abs(u), axis=0) <= 1e-04) == p:
                print("Stop! The value of mu1 is too large")
                break
            subiter_u += 1
        loading_trace_u[:, iter-1] = what_cut.ravel(order='F')
        subiter_v = 1
        dis_v = 10
        while dis_v > eps and subiter_v <= submaxstep:
            v_old = v.copy()
            if pen1 == "homogeneity":
                v = v_value_homo(u, v, p, q, L, mu1=mu3, mu2=mu4, pen2=pen2)
            if pen1 == "heterogeneity":
                v = v_value_hetero(u, v, p, q, L, mu1=mu3, mu2=mu4, pen2=pen2)
            v_norm = np.sqrt(np.sum(v**2, axis=0))
            v_norm[v_norm == 0] = 1e-04
            v = (v.T / np.expand_dims(v_norm, axis=-1)).T
            dis_v = np.max(np.sqrt(np.sum((v - v_old)**2, axis=0)) / np.sqrt(np.sum(v_old**2, axis=0)))
            what_v = v
            what_cut_v = np.where(np.abs(what_v) > 1e-04, what_v, 0)
            what_cut_norm = np.sqrt(np.sum(what_cut_v**2, axis=0))
            what_cut_norm[what_cut_norm == 0] = 1e-04
            what_dir = (what_cut_v.T / np.expand_dims(what_cut_norm, axis=-1)).T
            what_cut_v = np.where(np.abs(what_dir) > 1e-04, what_dir, 0)
            if np.sum(np.sum(np.abs(v), axis=0) <= 1e-04) == q:
                print("Stop! The value of mu1 is too large")
                break
            subiter_v += 1
        loading_trace_v[:, iter-1] = what_cut_v.ravel(order='F')
        dis_u_iter = np.max(np.sqrt(np.sum((u - u_iter)**2, axis=0)) / np.sqrt(np.sum(u_iter**2, axis=0)))
        dis_v_iter = np.max(np.sqrt(np.sum((v - v_iter)**2, axis=0)) / np.sqrt(np.sum(v_iter**2, axis=0)))
        if np.sum(np.sum(np.abs(u), axis=0) <= 1e-04) == p or np.sum(np.sum(np.abs(v), axis=0) <= 1e-04) == q:
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
            new2A_v = [np.where(what_cut_v[:, l] != 0)[0] for l in range(L)]
            for l in range(L):
                new2A_l = new2A_v[l]
                if len(new2A_l) <= 10:
                    print("DataSet {}:".format(l+1))
                    print(', '.join(['Y{}'.format(i+1) for i in new2A_l]))
                else:
                    print("DataSet {}:".format(l+1))
                    nlines = int(np.ceil(len(new2A_l) / 10))
                    for i in range(nlines - 1):
                        print(', '.join(['Y{}'.format(i+1) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                    print(', '.join(['Y{}'.format(i+1) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
        iter += 1
    if iter > 1:
        loading_trace_u = loading_trace_u[:, :iter-1]
        loading_trace_v = loading_trace_v[:, :iter-1]
    what_u = what_cut
    what_v = what_cut_v
    new2A = [ip[what_u[:, l] != 0] for l in range(L)]
    new2A_v = [iq[what_v[:, l] != 0] for l in range(L)]
    def plot_draw(draw):
        if draw:
            for l in range(L):
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(np.transpose(loading_trace_u[l * p : (l + 1) * p, ]))
                plt.title(f'Dataset {l+1}\nConvergence path of elements in vector u')
                plt.xlabel('Number of iterations')
                plt.ylabel('Weight')
                plt.subplot(1, 2, 2)
                plt.plot(np.transpose(loading_trace_v[l * q : (l + 1) * q, ]))
                plt.title(f'Dataset {l+1}\nConvergence path of elements in vector v')
                plt.xlabel('Number of iterations')
                plt.ylabel('Weight')
                plt.tight_layout()
                plt.show()
    plot_draw(draw = draw)  
    dictname = [f'Dataset {l+1}' for l in range(L)]
    new2A_dict = {}
    new2A_v_dict = {}
    meanx_dict = {}
    meany_dict = {}
    normx_dict = {}
    normy_dict = {}
    x_dict = {}
    y_dict = {}    
    for l in range(L):
        new2A_dict.update({dictname[l]:new2A[l]})
        new2A_v_dict.update({dictname[l]:new2A_v[l]})
        meanx_dict.update({dictname[l]:meanx[l]})
        meany_dict.update({dictname[l]:meany[l]})
        normx_dict.update({dictname[l]:normx[l]})
        normy_dict.update({dictname[l]:normy[l]})        
        x_dict.update({dictname[l]:x[l]})
        y_dict.update({dictname[l]:y[l]}) 
    what_df = pd.DataFrame(what, index = [str(i+1) for i in range(p)], columns = dictname)
    what_v_df = pd.DataFrame(what_v, index = [str(i+1) for i in range(p)], columns = dictname)
    class iscca:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.loading_x = data['loading.x']
            self.loading_y = data['loading.y']
            self.variable_x = data['variable.x']
            self.variable_y = data['variable.y']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.meany = data['meany']
            self.normy = data['normy']
            self.pen1 = data['pen1']
            self.pen2 = data['pen2']
            self.mu1 = data['mu1']
            self.mu2 = data['mu2']
            self.mu3 = data['mu3']
            self.mu4 = data['mu4']
            self.loading_trace_u = data['loading_trace_u']
            self.loading_trace_v = data['loading_trace_v']
    object_dict = {
        'x': x_dict, 
        'y': y_dict, 
        'loading.x': what_df, 
        'loading.y': what_v_df, 
        'variable.x': new2A_dict, 
        'variable.y': new2A_v_dict, 
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
        'pen1': pen1, 
        'pen2': pen2, 
        'mu1': mu1, 
        'mu2': mu2, 
        'mu3': mu3, 
        'mu4': mu4, 
        'loading_trace_u': loading_trace_u, 
        'loading_trace_v': loading_trace_v
    }
    object = iscca(object_dict)
    return object