import numpy as np
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds

def spca(x, mu1, eps=1e-04, scale_x=True, maxstep=50, trace=False):
    x = np.matrix(x)
    n, p = x.shape
    ip = np.arange(1,p+1)
    one = np.ones((1, n))
    meanx = one.dot(x) / n
    x = scale(x, with_std=False)
    if scale_x:
        normx = np.sqrt(np.dot(one, x**2) / (n - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some of the columns of the predictor matrix have zero variance.")
        x = scale(x, with_std=False) / normx
    else:
        normx = np.ones(p)
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
    what = np.zeros((p, 1))
    u, s, vt = svds(x, k=1)
    U = vt.T * s[0]
    V = u
    u = U
    v = V
    iter = 1
    dis_u = 10
    loading_trace = np.zeros((p, maxstep))
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis_u > eps and iter <= maxstep:
        u_old = u.copy()
        v_old = v.copy()
        s = np.sum(x * np.tile(v, p).reshape(n, p), axis=0) / n
        ro_d = [ro_d1st(s[j], mu1, 6) for j in range(p)]
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        u = np.array([fun_c(j) for j in range(p)])
        if np.sum(np.abs(u) <= 1e-04) == p:
            print("The value of mu1 is too large")
            what_cut = u
            break
        v = np.expand_dims(x.dot(u) / np.sqrt(np.sum((x.dot(u))**2)),axis=-1)
        u_norm = np.sqrt(np.sum(u**2))
        u_norm = 1e-04 if u_norm == 0 else u_norm
        u_scale = u / u_norm
        dis_u = np.sqrt(np.sum((u - u_old)**2)) / np.sqrt(np.sum(u_old**2))
        what = u_scale
        what_cut = np.where(np.abs(what) > 1e-04, what, 0)
        what_cut_norm = np.sqrt(np.sum(what_cut**2))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = what_cut / what_cut_norm
        what_cut = np.where(np.abs(what_dir) > 1e-04, what_dir, 0)
        loading_trace[:, iter-1] = what_cut
        if np.sum(np.abs(u_scale) <= 1e-04) == p:
            print("The value of mu1 is too large")
            break
        if trace:
            new2A = ip[what_cut != 0]
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
    loading_trace = loading_trace[:, :iter-2]
    what = what_cut
    new2A = ip[what != 0]
    eigenvalue = what.T.dot(np.cov(x, rowvar = False)).dot(what)
    comp = x.dot(what)
    class spca:
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
        'x': x, 
        'eigenvalue': eigenvalue, 
        'eigenvector': what, 
        'component': comp, 
        'variable': new2A, 
        'meanx': meanx,
        'normx': normx, 
        'mu1': mu1, 
    }
    object = spca(object_dict)
    return object