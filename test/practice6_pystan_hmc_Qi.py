# HMC estimation of the new model
# Simulated data
import numpy as np
np.random.seed(0)
import numpy.random as npr
from pyglm.utils.utils import expand_scalar, compute_optimal_rotation
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
plt.ion()

dim = 2
N = 10
r = 2 + 2 * (np.arange(N) // (N/2.))
th = np.linspace(0, 4 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L = np.hstack((x[:, None], y[:, None]))

# Weight model
W = np.zeros((N, N))
# Distance matrix
D = ((L[:, None, :] - L[None, :, :]) ** 2).sum(2)
sig = np.exp(-D/2)
Sig = np.tile(sig[:, :, None, None], (1, 1, 1, 1))

Mu = expand_scalar(0, (N, N, 1))

for n in range(N):
    for m in range(N):
        W[n, m] = npr.multivariate_normal(Mu[n, m], Sig[n, m])

# Adjacency model
fig, axs = plt.subplots(4,5)
P = [0.1, 0.3, 0.5, 0.7]
i0 = 3

import pickle
sm = pickle.load(open('/Users/pillowlab/Dropbox/pyglm-master/Practices/model2.pkl', 'rb'))

A = 1 * (np.random.rand(N, N) < P[i0])

W = A * W

new_data = dict(N=N, W=W, A=A)
fit = sm.sampling(data=new_data, iter=1000, chains=4)

samples = fit.extract(permuted=True)
L_estimate_all = samples['l']
p_estimate_all = samples['p']
eta_estimate_all = samples['eta']
rho_estimate_all = samples['rho']

for i in range(2000):
    R = compute_optimal_rotation(L_estimate_all[i, :, :], L)
    L_estimate_all[i, :, :] = np.dot(L_estimate_all[i, :, :], R)

L_estimate = np.mean(L_estimate_all, 0)
sns.heatmap(W, ax=axs[i0, 0])
sns.heatmap(A, ax=axs[i0, 1])
sns.kdeplot(samples['p'], ax=axs[i0, 2])
axs[i0, 2].vlines(P[i0], 0, 10, colors="r", linestyles="dashed")
axs[i0, 3].scatter(L[:, 0], L[:, 1])

from hips.plotting.colormaps import harvard_colors

color = harvard_colors()[0:10]
for i in range(N):
    axs[i0, 4].scatter(L_estimate_all[-50:, i, 0], L_estimate_all[-50:, i, 1], c=color[i], s=10)
