import numpy as np
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds

def meta_scca(x, y, L, mu1, mu2, eps=1e-04, scale_x=True, scale_y=True, maxstep=50, trace=False):
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
    def fun_1(l, x, y):
        return np.dot(np.transpose(x[l]), y[l])
    zz = [fun_1(l, x, y) for l in range(L)]
    z = np.zeros((p, q))
    for l in range(L):
        z += zz[l]
    z /= (np.sum(nl) - L)
    u, s, vt = svds(z, k=1)
    U = u
    V = vt.T
    u = U
    v = V
    iter = 1
    dis_u = 10
    dis_v = 10
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis_u > eps and dis_v > eps and iter <= maxstep:
        u_old = np.copy(u)
        v_old = np.copy(v)
        s = np.array(np.dot(v.T, z.T)).reshape(-1)
        ro_d = [ro_d1st(s[j], mu1, 6) for j in range(p)]
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        u = np.array([fun_c(j) for j in range(p)])
        u_norm = np.sqrt(np.sum(np.square(u)))
        u_norm = 1e-04 if u_norm == 0 else u_norm
        u = u / u_norm
        dis_u = np.sqrt(np.sum(np.square(u - u_old))) / np.sqrt(np.sum(np.square(u_old)))
        what = u
        what_cut = np.where(abs(what) > 1e-04, what, 0)
        what_cut_norm = np.sqrt(np.sum(np.square(what_cut)))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = what_cut / what_cut_norm
        what_cut = np.where(abs(what_dir) > 1e-04, what_dir, 0)
        s = np.dot(u.T, z)
        ro_d = [ro_d1st(s[j], mu2, 6) for j in range(q)]
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        v = np.array([fun_c(j) for j in range(q)])
        v_norm = np.sqrt(np.sum(np.square(v)))
        v_norm = 1e-04 if v_norm == 0 else v_norm
        v = v / v_norm
        dis_v = np.sqrt(np.sum(np.square(v - v_old))) / np.sqrt(np.sum(np.square(v_old)))
        what_v = v
        what_cut_v = np.where(abs(what_v) > 1e-04, what_v, 0)
        what_cut_norm = np.sqrt(np.sum(np.square(what_cut_v)))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = what_cut_v / what_cut_norm
        what_cut_v = np.where(abs(what_dir) > 1e-04, what_dir, 0)
        dis_u = np.sqrt(np.sum(np.square(u - u_old))) / np.sqrt(np.sum(np.square(u_old)))
        dis_v = np.sqrt(np.sum(np.square(v - v_old))) / np.sqrt(np.sum(np.square(v_old)))
        if np.sum(abs(v) <= 1e-04) == q or np.sum(abs(u) <= 1e-04) == p:
            break
        if trace:
            new2A_u = ip[what_cut != 0]
            new2A_v = iq[what_cut_v != 0]
            print("\n--------------------\n----- Step {} -----\n--------------------".format(iter))
            new2A_l = new2A_u
            if len(new2A_l) <= 10:
                print(', '.join(['X{}'.format(i) for i in new2A_l]))
            else:
                nlines = int(np.ceil(len(new2A_l) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
            new2A_l = new2A_v
            if len(new2A_l) <= 10:
                print(', '.join(['Y{}'.format(i) for i in new2A_l]))
            else:
                nlines = int(np.ceil(len(new2A_l) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['Y{}'.format(i) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['Y{}'.format(i) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
        iter += 1
    what = what_cut
    what_v = what_cut_v
    new2A = ip[what != 0]
    new2A_v = iq[what_v != 0]
    dictname = [f'Dataset {l+1}' for l in range(L)]
    meanx_dict = {}
    meany_dict = {}
    normx_dict = {}
    normy_dict = {}
    x_dict = {}
    y_dict = {}    
    for l in range(L):
        meanx_dict.update({dictname[l]:meanx[l]})
        meany_dict.update({dictname[l]:meany[l]})
        normx_dict.update({dictname[l]:normx[l]})
        normy_dict.update({dictname[l]:normy[l]})        
        x_dict.update({dictname[l]:x[l]})
        y_dict.update({dictname[l]:y[l]}) 
    class meta_scca:
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
            self.mu1 = data['mu1']
            self.mu2 = data['mu2']
    object_dict = {
        'x': x_dict, 
        'y': y_dict, 
        'loading.x': what, 
        'loading.y': what_v, 
        'variable.x': new2A, 
        'variable.y': new2A_v, 
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
        'mu1': mu1, 
        'mu2': mu2, 
    }
    object = meta_scca(object_dict)
    return object
