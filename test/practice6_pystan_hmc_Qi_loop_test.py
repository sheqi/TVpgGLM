import pickle
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from pyglm.utils.utils import expand_scalar, compute_optimal_rotation

dim = 2
N = 20
r = 1 + np.arange(N) // (N/2.)
th = np.linspace(0, 4 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L = np.hstack((x[:, None], y[:, None]))

L1 = np.random.randn(N, dim)

W = np.zeros((N, N))
# Distance matrix
D = ((L[:, None, :] - L[None, :, :]) ** 2).sum(2)
sig = np.exp(-D/2)
Sig = np.tile(sig[:, :, None, None], (1, 1, 1, 1))

Mu = expand_scalar(0, (N, N, 1))

for n in range(N):
    for m in range(N):
        W[n, m] = npr.multivariate_normal(Mu[n, m], Sig[n, m])

aa = 1.0
bb = 1.0
cc = 1.0

sm = pickle.load(open('/Users/pillowlab/Dropbox/pyglm-master/Practices/model.pkl', 'rb'))

new_data = dict(N=N, W=W, B=dim)

for i in range(100):
    fit = sm.sampling(data=new_data, iter=100, warmup=50, chains=1, init=[dict(l=L1, sigma=aa)],
                      control=dict(stepsize=0.001))

    samples = fit.extract(permuted=True)
    aa = np.mean(samples['sigma'])
    #aa = samples['sigma'][-1]
    #bb = np.mean(samples['eta'])
    #cc = np.mean(samples['rho'])
    L1 = np.mean(samples['l'], 0)
    #L1 = samples['l'][-1]
    R = compute_optimal_rotation(L1, L)
    L1 = np.dot(L1, R)

plt.scatter(L1[:,0],L1[:,1])
plt.scatter(L[:,0],L[:,1])