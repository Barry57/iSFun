import numpy as np
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds

def meta_spca(x, L, mu1, eps=1e-04, scale_x=True, maxstep=50, trace=False):
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
    z = np.zeros((p, p))
    for l in range(L):
        z += (nl[l] - 1) * np.cov(x[l], rowvar = False)
    z /= (np.sum(nl) - L)
    u, s, vt = svds(z, k=1)
    v = u
    lambda_ = np.zeros(p)
    dis_u = 10
    iter = 1
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis_u > eps and iter <= maxstep:
        u_old = np.copy(u)
        v_old = np.copy(v)
        s = np.array(np.dot(v.T, z)).reshape(-1)
        ro_d = [ro_d1st(s[j], mu1, 6) for j in range(p)]
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        u = np.array([fun_c(j) for j in range(p)])
        if np.sum(abs(u) <= 1e-04) == p:
            print("The value of mu1 is too large")
            break
        v = np.dot(z, u) / np.sqrt(np.sum(np.square(np.dot(z, u))))
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
        if np.sum(abs(u) <= 1e-04) == p:
            print("The value of mu1 is too large")
            break
        if trace:
            new2A = ip[what != 0]
            print("\n--------------------\n----- Step {} -----\n--------------------".format(iter))
            new2A_l = new2A
            if len(new2A_l) <= 10:
                print(', '.join(['X{}'.format(i) for i in new2A_l]))
            else:
                nlines = int(np.ceil(len(new2A_l) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
        iter += 1
    what = what_cut
    new2A = ip[what != 0]
    eigenvalue = [what.T.dot(np.cov(x[l], rowvar = False)).dot(what) for l in range(L)]
    comp = [x[l].dot(what) for l in range(L)]
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
    class meta_spca:
        def __init__(self, data):
            self.x = data['x']
            self.eigenvalue = data['eigenvalue']
            self.eigenvector = data['eigenvector']
            self.component = data['component']
            self.variable = data['variable']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.mu1 = data['mu1']
    object_dict = {
        'x': x_dict, 
        'eigenvalue': eigenvalue_dict, 
        'eigenvector': what, 
        'component': comp_dict, 
        'variable': new2A, 
        'meanx': meanx_dict,
        'normx': normx_dict, 
        'mu1': mu1, 
    }
    object = meta_spca(object_dict)
    return object