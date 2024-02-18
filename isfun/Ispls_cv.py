import numpy as np
from scipy.integrate import quad
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from numpy.linalg import svd
from numpy.linalg import solve
from numpy.linalg import norm
from scipy.optimize import fsolve
from numpy.linalg import lstsq
import itertools
from .Ispls import ispls

def ispls_cv(x, y, L, K, mu1, mu2, eps=1e-04, kappa=0.05, pen1="homogeneity", 
             pen2="magnitude", scale_x=True, scale_y=True, maxstep=50, submaxstep=10):
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
    def c_value_homo(z, a, c, p, q, L, mu1, mu2, pen2, nl):
        def fun_s(j, l):
            z_l = z[:, (l * q):(l + 1) * q]
            a_l = a[:, l].reshape(-1, 1)
            c_j = c[j, :]
            s1 = np.dot(a_l.T, np.dot(z_l, z_l[j, :].reshape(1, -1).T)) / (nl[l]**2)
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(c_j[np.arange(len(c_j))!=l])
            elif pen2 == "sign":
                s2 = mu2 * np.sum(np.array([x / np.sqrt(x**2 + 0.5) for x in c_j[np.arange(len(c_j))!=l]])) / np.sqrt(c_j[l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        norm_c_j = np.sqrt(np.sum(c**2, axis=1))
        ro_d = ro_d1st(norm_c_j, mu1, 6)
        def fun_c(j, l):
            s_norm = np.sqrt(np.sum(result_s[j, :]**2))
            if pen2 == "magnitude":
                temp = (s_norm > ro_d[j]) * result_s[j, l] * (s_norm - ro_d[j]) / ((1/q/(nl[l]**2) + mu2 * (L - 1)) * s_norm)
            elif pen2 == "sign":
                temp = (s_norm > ro_d[j]) * result_s[j, l] * (s_norm - ro_d[j]) / ((1/q/(nl[l]**2) + mu2 * (L - 1)/(c[j, l]**2 + 0.5)) * s_norm)
            return temp
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    def c_value_hetero(z, a, c, p, q, L, mu1, mu2, pen2, nl):
        def fun_mu(j, l):
            c_j = c[j, :]
            ro_j = ro(np.abs(c_j), mu1, 6)
            s_ro = np.sum(ro_j)
            mu_jl = ro_d1st(s_ro, 1, 0.5 * L * 6 * mu1**2) * ro_d1st(np.abs(c_j[l]), mu1, 6)
            return mu_jl
        result_mu = np.array([fun_mu(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        def fun_s(j, l):
            z_l = z[:, (l * q):(l + 1) * q]
            a_l = a[:, l].reshape(-1, 1)
            c_j = c[j, :]
            s1 = np.dot(a_l.T, np.dot(z_l, z_l[j, :].reshape(1, -1).T)) / (nl[l]**2)
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(c_j[np.arange(len(c_j))!=l])
            elif pen2 == "sign":
                s2 = mu2 * np.sum(np.array([x / np.sqrt(x**2 + 0.5) for x in c_j[np.arange(len(c_j))!=l]])) / np.sqrt(c_j[l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        def fun_c(j, l):
            if pen2 == "magnitude":
                temp = np.sign(result_s[j, l]) * (np.abs(result_s[j, l]) > result_mu[j, l]) * (np.abs(result_s[j, l]) - result_mu[j, l]) / (1/q/(nl[l]**2) + mu2 * (L - 1))
            elif pen2 == "sign":
                temp = np.sign(result_s[j, l]) * (np.abs(result_s[j, l]) > result_mu[j, l]) * (np.abs(result_s[j, l]) - result_mu[j, l]) / (1/q/(nl[l]**2) + mu2 * (L - 1)/(c[j, l]**2 + 0.5))
            return temp
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    def c_value_homo_q1(z, a, c, p, q, L, mu1, mu2, pen2, nl):
        def fun_s(j, l):
            z_l = z[:, (l * q):(l + 1) * q]
            a_l = a[:, l].reshape(-1, 1)
            c_j = c[j, :]
            s1 = np.dot(a_l.T, np.dot(z_l, z_l[j].reshape(1, -1).T)) / (nl[l]**2)
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(c_j[np.arange(len(c_j))!=l])
            elif pen2 == "sign":
                s2 = mu2 * np.sum(np.array([x / np.sqrt(x**2 + 0.5) for x in c_j[np.arange(len(c_j))!=l]])) / np.sqrt(c_j[l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        norm_c_j = np.sqrt(np.sum(c**2, axis=1))
        ro_d = ro_d1st(norm_c_j, mu1, 6)
        def fun_c(j, l):
            s_norm = np.sqrt(np.sum(result_s[j, :]**2))
            if pen2 == "magnitude":
                temp = (s_norm > ro_d[j]) * result_s[j, l] * (s_norm - ro_d[j]) / ((1/q/(nl[l]**2) + mu2 * (L - 1)) * s_norm)
            elif pen2 == "sign":
                temp = (s_norm > ro_d[j]) * result_s[j, l] * (s_norm - ro_d[j]) / ((1/q/(nl[l]**2) + mu2 * (L - 1)/(c[j, l]**2 + 0.5)) * s_norm)
            return temp
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        return c
    def c_value_hetero_q1(z, a, c, p, q, L, mu1, mu2, pen2, nl):
        def fun_mu(j, l):
            c_j = c[j, :]
            ro_j = ro(np.abs(c_j), mu1, 6)
            s_ro = np.sum(ro_j)
            mu_jl = ro_d1st(s_ro, 1, 0.5 * L * 6 * mu1**2) * ro_d1st(np.abs(c_j[l]), mu1, 6)
            return mu_jl
        result_mu = np.array([fun_mu(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        def fun_s(j, l):
            z_l = z[:, (l * q):(l + 1) * q]
            a_l = a[:, l].reshape(-1, 1)
            c_j = c[j, :]
            s1 = np.dot(a_l.T, np.dot(z_l, z_l[j].reshape(1, -1).T)) / (nl[l]**2)
            if pen2 == "magnitude":
                s2 = mu2 * np.sum(c_j[np.arange(len(c_j))!=l])
            elif pen2 == "sign":
                s2 = mu2 * np.sum(np.array([x / np.sqrt(x**2 + 0.5) for x in c_j[np.arange(len(c_j))!=l]])) / np.sqrt(c_j[l]**2 + 0.5)
            s = s1 + s2
            return s
        result_s = np.array([fun_s(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
        def fun_c(j, l):
            if pen2 == "magnitude":
                temp = np.sign(result_s[j, l]) * (np.abs(result_s[j, l]) > result_mu[j, l]) * (np.abs(result_s[j, l]) - result_mu[j, l]) / (1/q/(nl[l]**2) + mu2 * (L - 1))
            elif pen2 == "sign":
                temp = np.sign(result_s[j, l]) * (np.abs(result_s[j, l]) > result_mu[j, l]) * (np.abs(result_s[j, l]) - result_mu[j, l]) / (1/q/(nl[l]**2) + mu2 * (L - 1)/(c[j, l]**2 + 0.5))
            return temp
        c = np.array([fun_c(j, l) for j in range(p) for l in range(L)]).reshape(p, L)
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
            z_l = np.dot(np.transpose(x_train[l]), y_train[l])
            z_l = z_l / nl_train[l]
            return z_l
        z = np.hstack(np.array([fun_1(l) for l in range(L)]))
        def fun_2(l):
            z_l = z[:, (l * q):(l + 1) * q]
            m_l = np.dot(z_l, z_l.T) / q
            return m_l
        m = np.hstack(np.array([fun_2(l) for l in range(L)]))
        dis = 10
        iter = 1
        kappa2 = (1 - kappa) / (1 - 2 * kappa)
        def fun_c(l):
            return svd(np.dot(z[:, (l * q):(l + 1) * q], z[:, (l * q):(l + 1) * q].T), full_matrices=False)[0][:, 0]
        c = np.array([fun_c(l) for l in range(L)]).T
        sgn1 = np.array([np.sign(np.dot(np.dot(c[:, l].T, z[:, (l * q):(l + 1) * q]), np.dot(z[:, (l * q):(l + 1) * q].T, c[:, l]))) for l in range(L)])
        for l in range(L):
            c[:, l] = sgn1[l] * c[:, l]
        sgn2 = np.array([np.sign(np.dot(c[:, 0].T, c[:, l])) / (np.sqrt(np.sum(c[:, 0]**2)) * np.sqrt(np.sum(c[:, l]**2))) for l in range(1, L)])
        for l in range(1, L):
            c[:, l] = sgn2[l - 1] * c[:, l]
        a = c
        dis_c_iter = 10
        while dis_c_iter > eps and iter <= maxstep:
            c_iter = c.copy()
            def fun_3(l):
                def h(lambda_):
                    alpha = solve(m[:, l * p: (l + 1) * p] + lambda_ * np.eye(p), np.dot(m[:, l * p: (l + 1) * p], c[:, l]))
                    obj = np.dot(alpha.T, alpha) - 1 / kappa2**2
                    return obj
                while h(1e-04) * h(1e+12) > 0:
                    m[:, (l - 1) * p: l * p] *= 2
                    c[:, l] *= 2
                lambdas = fsolve(h, x0=1e-04)
                a_l = kappa2 * solve(m[:, l * p: (l + 1) * p] + lambdas * np.eye(p), np.dot(m[:, l * p: (l + 1) * p], c[:, l]))
                return a_l
            a = np.array([fun_3(l) for l in range(L)]).T
            subiter_c = 1
            dis_c = 10
            while dis_c > eps and subiter_c <= submaxstep:
                c_old = c.copy()
                if pen1 == "homogeneity" and q != 1:
                    c = c_value_homo(z, a, c, p, q, L, mu1, mu2, pen2, nl = nl_train)
                elif pen1 == "homogeneity" and q == 1:
                    c = c_value_homo_q1(z, a, c, p, q, L, mu1, mu2, pen2, nl = nl_train)
                elif pen1 == "heterogeneity" and q != 1:
                    c = c_value_hetero(z, a, c, p, q, L, mu1, mu2, pen2, nl = nl_train)
                elif pen1 == "heterogeneity" and q == 1:
                    c = c_value_hetero_q1(z, a, c, p, q, L, mu1, mu2, pen2, nl = nl_train)
                c_norm = np.sqrt(np.sum(c**2, axis=0))
                c_norm[c_norm == 0] = 1e-04
                c = (c.T / np.expand_dims(c_norm,axis=-1)).T
                dis = np.max(np.sqrt(np.sum((c - c_old)**2, axis=0)) / np.sqrt(np.sum(c_old**2, axis=0)))
                what = c.copy()
                what_cut = np.where(np.abs(what) > 1e-04, what, 0)
                what_cut_norm = np.sqrt(np.sum(what_cut**2, axis=0))
                what_cut_norm[what_cut_norm == 0] = 1e-04
                what_dir = (what_cut.T / np.expand_dims(what_cut_norm, axis=-1)).T
                what_cut = np.where(np.abs(what_dir) > 1e-04, what_dir, 0)
                if np.sum(np.apply_along_axis(lambda x: np.sum(np.abs(x) <= 1e-04) == p, axis=0, arr=c)) > 0:
                    print("Stop! The value of mu1 is too large")
                    break
                subiter_c += 1
            dis_c_iter = np.max(np.sqrt(np.sum((c - c_iter)**2, axis=0)) / np.sqrt(np.sum(c_iter**2, axis=0)))
            if np.sum(np.apply_along_axis(lambda x: np.sum(np.abs(x) <= 1e-04) == p, axis=0, arr=c)) > 0:
                break
            iter += 1
        def fun_fit_cv(l):
            x_l = x_train[l]
            w_l = what[:, l].reshape(-1, 1)
            t_l = np.dot(x_l, w_l)
            if np.sum(w_l == 0) != p:
                y_l = y_train[l]
                fit_l = lstsq(t_l, y_l, rcond=None)[0]
                betahat_l = np.dot(w_l, fit_l).reshape(p, q)
            else:
                betahat_l = np.zeros((p, q))
            return betahat_l
        betahat_cv = [fun_fit_cv(i) for i in range(L)]
        def get_rho(l):
            return norm(y_test[l] - np.matmul(x_test[l], betahat_cv[l]))**2
        cv_rho = sum([get_rho(l) for l in range(L)])
        return cv_rho
    mu_list = [mu1, mu2]
    for i in range(len(mu_list)):
        if not isinstance(mu_list[i], list):
            mu_list[i] = [mu_list[i]]
    mu1, mu2 = mu_list
    mu = np.array(list(itertools.product(mu1, mu2)))
    for i in range(len(mu_list)):
        if isinstance(mu_list[i], list) and len(mu_list[i]) > 1:
            mu_list[i] = np.array(mu_list[i])
        else:
            mu_list[i] = float(mu_list[i][0])
    mu1, mu2 = mu_list
    rho = []
    for loop in range(mu.shape[0]):
        mu1 = mu[loop, 0]
        mu2 = mu[loop, 1]
        cv_fun_list = []
        for k in range(K):
            cv_fun_list.append(cv_fun(k))
        rho.append(np.mean(cv_fun_list))
    index = np.argmax(rho)
    mu1_final = mu[index, 0]
    mu2_final = mu[index, 1]
    result = ispls(x, y, L, mu1 = mu1_final, mu2 = mu2_final, eps=eps, kappa = kappa, 
                   pen1 = pen1, pen2 = pen2, scale_x = scale_x, scale_y = scale_y, 
                   maxstep = maxstep, trace = False, draw = False)
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
    class ispls_cv:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.mu1 = data['mu1']
            self.mu2 = data['mu2']
            self.fold = data['fold']
            self.betahat = data['betahat']
            self.loading = data['loading']
            self.variable = data['variable']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.meany = data['meany']
            self.normy = data['normy']
            self.pen1 = data['pen1']
            self.pen2 = data['pen2']
            self.kappa = data['kappa']
            self.loading_trace = data['loading_trace']
    object_dict = {
        'x': x_dict, 
        'y': y_dict,
        'mu1': mu1_final, 
        'mu2': mu2_final, 
        'fold': folds_dict,
        'betahat': result.betahat,
        'loading': result.loading,
        'variable': result.variable, 
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
        'pen1': pen1, 
        'pen2': pen2, 
        'kappa': kappa, 
        'loading_trace': result.loading_trace
    }
    object = ispls_cv(object_dict)
    return object