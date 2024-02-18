import numpy as np
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold
import itertools
from .Iscca import iscca

def iscca_cv(x, y, L, mu1, mu2, mu3, mu4, K=5, eps=1e-04, pen1="homogeneity", 
          pen2="magnitude", scale_x=True, scale_y=True, maxstep=50, 
          submaxstep=10):
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
    folds = [list(KFold(n_splits=K, shuffle = True).split(range(nl[l]))) for l in range(L)]
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
    def u_value_homo(u, v, p, q, L, mu1, mu2, pen2, x, y, nl):
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
    def u_value_hetero(u, v, p, q, L, mu1, mu2, pen2, x, y, nl):
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
    def v_value_homo(u, v, p, q, L, mu1, mu2, pen2, x, y, nl):
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
    def v_value_hetero(u, v, p, q, L, mu1, mu2, pen2, x, y, nl):
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
    def fun_k(k):
        x_train = [x[l][folds[l][k][0]] for l in range(L)]
        y_train = [y[l][folds[l][k][0]] for l in range(L)]
        x_test = [x[l][folds[l][k][1]] for l in range(L)]
        y_test = [y[l][folds[l][k][1]] for l in range(L)]
        nl_train = [len(x_t) for x_t in x_train]
        nl_test = [len(x_t) for x_t in x_test]
        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 
                'y_test': y_test, 'nl_train': nl_train, 'nl_test': nl_test}
    data_cv = [fun_k(k) for k in range(K)]
    def cv_fun(k):
        x_train = data_cv[k]['x_train']
        y_train = data_cv[k]['y_train']
        x_test = data_cv[k]['x_test']
        y_test = data_cv[k]['y_test']
        nl_train = data_cv[k]['nl_train']
        nl_test = data_cv[k]['nl_test']
        what = np.zeros((p, L))
        def fun_1(l):
            Z_l = svds(np.dot(x_train[l].T, y_train[l]), k=1)
            u_l = Z_l[0]
            return u_l
        U = np.hstack(np.array([fun_1(l) for l in range(L)]))
        def fun_2(l):
             Z_l = svds(np.dot(x_train[l].T, y_train[l]), k=1)
             v_l = Z_l[2]
             return v_l
        V = np.vstack(np.array([fun_2(l) for l in range(L)])).T
        sgn1 = [np.sign(np.dot(np.dot(U[:, l], x_train[l].T), np.dot(y_train[l], V[:, l]))) for l in range(L)]
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
        while dis_u_iter > eps and dis_v_iter > eps and iter <= maxstep:
            u_iter = u.copy()
            v_iter = v.copy()
            subiter_u = 1
            dis_u = 10
            while dis_u > eps and subiter_u <= submaxstep:
                u_old = u.copy()
                if pen1 == "homogeneity":
                    u = u_value_homo(u, v, p, q, L, mu1, mu2, pen2, x = x_train, y = y_train, nl = nl_train)
                if pen1 == "heterogeneity":
                    u = u_value_hetero(u, v, p, q, L, mu1, mu2, pen2, x = x_train, y = y_train, nl = nl_train)
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
            subiter_v = 1
            dis_v = 10
            while dis_v > eps and subiter_v <= submaxstep:
                v_old = v.copy()
                if pen1 == "homogeneity":
                    v = v_value_homo(u, v, p, q, L, mu1=mu3, mu2=mu4, pen2=pen2, x = x_train, y = y_train, nl = nl_train)
                if pen1 == "heterogeneity":
                    v = v_value_hetero(u, v, p, q, L, mu1=mu3, mu2=mu4, pen2=pen2, x = x_train, y = y_train, nl = nl_train)
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
                dis_u_iter = np.max(np.sqrt(np.sum((u - u_iter)**2, axis=0)) / np.sqrt(np.sum(u_iter**2, axis=0)))
                dis_v_iter = np.max(np.sqrt(np.sum((v - v_iter)**2, axis=0)) / np.sqrt(np.sum(v_iter**2, axis=0)))
                if np.sum(np.sum(np.abs(u), axis=0) <= 1e-04) == p or np.sum(np.sum(np.abs(v), axis=0) <= 1e-04) == q:
                    break
                iter += 1
        def fun_3(l):
            cor_a = np.dot(np.dot(np.dot(what_cut[:, l].T, x_test[l].T), y_test[l]), what_cut_v[:, l])
            cor_b = np.dot(np.dot(np.dot(what_cut[:, l].T, x_test[l].T), x_test[l]), what_cut[:, l])
            cor_c = np.dot(np.dot(np.dot(what_cut_v[:, l].T, y_test[l].T), y_test[l]), what_cut_v[:, l])
            if cor_b != 0 and cor_c != 0:
                rho_temp = cor_a / (np.sqrt(cor_b) * np.sqrt(cor_c))
            else:
                rho_temp = cor_a / (np.sqrt(cor_b) * np.sqrt(cor_c) + 1e-08)
            return rho_temp
        cv_rho = sum([abs(fun_3(i)) for i in range(L)])
        return cv_rho
    mu_list = [mu1, mu2, mu3, mu4]
    for i in range(len(mu_list)):
        if not isinstance(mu_list[i], list):
            mu_list[i] = [mu_list[i]]
    mu1, mu2, mu3, mu4 = mu_list
    mu = np.array(list(itertools.product(mu1, mu2, mu3, mu4)))
    for i in range(len(mu_list)):
        if isinstance(mu_list[i], list) and len(mu_list[i]) > 1:
            mu_list[i] = np.array(mu_list[i])
        else:
            mu_list[i] = float(mu_list[i][0])
    mu1, mu2, mu3, mu4 = mu_list
    rho = []
    for loop in range(mu.shape[0]):
        mu1 = mu[loop, 0]
        mu2 = mu[loop, 1]
        mu3 = mu[loop, 2]
        mu4 = mu[loop, 3]
        cv_fun_list = []
        for k in range(K):
            cv_fun_list.append(cv_fun(k))
        rho.append(np.mean(cv_fun_list))
    index = np.argmax(rho)
    mu1_final = mu[index, 0]
    mu2_final = mu[index, 1]
    mu3_final = mu[index, 2]
    mu4_final = mu[index, 3]   
    result = iscca(x, y, L, mu1 = mu1_final, mu2 = mu2_final, 
                    mu3 = mu3_final, mu4 = mu4_final, eps=eps, pen1 = pen1, 
                    pen2 = pen2, scale_x = scale_x, scale_y = scale_y, maxstep = maxstep, 
                    trace = False, draw = False)
    dictname = [f'Dataset {l+1}' for l in range(L)]
    folds_dict = {}
    meanx_dict = {}
    meany_dict = {}
    normx_dict = {}
    normy_dict = {}
    x_dict = {}
    y_dict = {}    
    for l in range(L):
        folds_dict.update({dictname[l]:folds[l]})
        meanx_dict.update({dictname[l]:meanx[l]})
        meany_dict.update({dictname[l]:meany[l]})
        normx_dict.update({dictname[l]:normx[l]})
        normy_dict.update({dictname[l]:normy[l]})        
        x_dict.update({dictname[l]:x[l]})
        y_dict.update({dictname[l]:y[l]})            
    class iscca_cv:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.mu1 = data['mu1']
            self.mu2 = data['mu2']
            self.mu3 = data['mu3']
            self.mu4 = data['mu4']
            self.fold = data['fold']
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
            self.loading_trace_u = data['loading_trace_u']
            self.loading_trace_v = data['loading_trace_v']
    object_dict = {
        'x': x_dict, 
        'y': y_dict,
        'mu1': mu1_final, 
        'mu2': mu2_final, 
        'mu3': mu3_final, 
        'mu4': mu4_final, 
        'fold': folds_dict,
        'loading.x': result.loading_x,
        'loading.y': result.loading_y,
        'variable.x': result.variable_x, 
        'variable.y': result.variable_y, 
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
        'pen1': pen1, 
        'pen2': pen2, 
        'loading_trace_u': result.loading_trace_u, 
        'loading_trace_v': result.loading_trace_v
    }
    object = iscca_cv(object_dict)
    return object
