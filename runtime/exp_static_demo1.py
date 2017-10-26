import sys;

sys.path.append("/Users/roger/Dropbox/pyglm-master/example")
sys.path.append("/Users/roger/Dropbox/pyglm-master/pyglm/")
sys.path.append("/Users/roger/Dropbox/pyglm-master/pyglm/utils/")

import numpy as np
import pickle
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
plt.ion()

from pybasicbayes.util.text import progprint_xrange
from utils.utils import compute_optimal_rotation
from pyglm.utils.basis import cosine_basis
from models import LatentDistanceWeightsSparseBernoulliGLM

# load experimental data
import scipy.io as sio
matfn = '/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/data/Result_WT_37C_Temporal_2008-10-13-1.mat'
data = sio.loadmat(matfn)

Y = data['DATA_sub']
L_true = data['D']
L_true_norm = np.zeros((28,2))
L_true_norm[:,0] = -1 + 2/(max(L_true[:,0])-min(L_true[:,0]))*(L_true[:,0]-min(L_true[:,0]))
L_true_norm[:,1] = -1 + 2/(max(L_true[:,1])-min(L_true[:,1]))*(L_true[:,1]-min(L_true[:,1]))

T = Y.shape[0]   # Number of time bins to generate
N = Y.shape[1]       # Number of neurons
B = 1       # Number of "basis functions"
L = 1     # Autoregressive window of influence

basis = cosine_basis(B=B, L=L) / L

# Create a test model for fitting
test_model = LatentDistanceWeightsSparseBernoulliGLM(N, basis=basis,
                        regression_kwargs=dict(S_w=0.5, mu_b=-2.))

test_model.add_data(Y)

# Fit with Gibbs sampling
def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0], m.network.L

def _update(m, itr):
    m.resample_model()
    return _collect(m)

N_samples = 500
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps, W_smpls, A_smpls, b_smpls, fr_smpls, L_smpls = tuple(map(np.array, samples))

# Plot the posterior mean and variance
W_mean = W_smpls[N_samples//2:].mean(0)
A_mean = A_smpls[N_samples//2:].mean(0)
fr_mean = fr_smpls[N_samples//2:].mean(0)
fr_std = fr_smpls[N_samples//2:].std(0)
L_mean = L_smpls[N_samples//2:].mean(0)

for i in range(N_samples):
    R = compute_optimal_rotation(L_smpls[i], L_true_norm)
    L_smpls[i] = L_smpls[i].dot(R)

# Mean location
D_est = np.sqrt((L_mean[:,None,:] - L_mean[None,:,:])**2).sum(2)

# Plot weights matrix across distance
D_mat = data['D_mat']
plt.scatter(D_mat.reshape((-1,1)),W_mean.reshape((-1,1)))
plt.figure()
plt.scatter(D_mat.reshape((-1,1)),D_est.reshape((-1,1)))

fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal")
k = 0
for i in range(N):
    ax.scatter(L_smpls[-50:, i, 0], L_smpls[-50:, i, 1], s=100, alpha=0.7, color=np.random.rand(3,1))
    k += 1

# cross-correlation
xc=np.corrcoef(Y.transpose())
plt.figure()
sns.heatmap(xc)
plt.figure()
sns.heatmap(W_mean[:,:,0])

# np.save('results/exp1.npy', L_smpls)

# Saving the objects:
with open('TVpgGLM/results/exp_static_electrode.pickle', 'wb') as f:
    pickle.dump([lps, W_smpls, A_smpls, b_smpls, L_smpls], f)
