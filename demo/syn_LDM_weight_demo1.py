import sys;

sys.path.append("/Users/roger/Dropbox/pyglm-master/pyglm/")
sys.path.append("/Users/roger/Dropbox/pyglm-master/pyglm/utils/")

import numpy as np
import pickle

np.random.seed(150)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
sns.set_context("paper")
plt.ion()

from pybasicbayes.util.text import progprint_xrange

from pyglm.utils.basis import cosine_basis
from pyglm.plotting import plot_glm
from models import LatentDistanceWeightsSparseBernoulliGLM

T = 10000  # Number of time bins to generate
N = 10  # Number of neurons
B = 1  # Number of "basis functions"
L = 100  # Autoregressive window of influence
D = 2  # Dimensionality of the feature space

# Create a cosine basis to model smooth influence of
# spikes on one neuron on the later spikes of others.
basis = cosine_basis(B=B, L=L) / L

true_model = LatentDistanceWeightsSparseBernoulliGLM(N, basis=basis, regression_kwargs=dict(rho=0.7, S_w=1, mu_b=-2))

# Set the true locations to be on a circle
# r = 1.5 + np.arange(N) // (N / 2.)
r = 1.55
th = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L_D = np.hstack((x[:, None], y[:, None]))

true_model.network.L = L_D

# Simulated weights
for m in range(N):
    for n in range(N):
        true_model.regressions[m].W[n, :] = true_model.regressions[m].a[n] * np.random.multivariate_normal(
            true_model.network.mu_W[m, n], true_model.network.sigma_W[m, n])

for k in range(N):
    true_model.regressions[k].a[k] = True
    true_model.regressions[k].W[k,:] = -3

_, Y = true_model.generate(T=T, keep=True)

mean_spikecount = Y.sum(0) / T

# Plot raster plot
sns.heatmap(np.transpose(Y), xticklabels=False)

# Plot the true model
fig, axs, handles = true_model.plot()
plt.pause(0.1)

# Plot cross-correlation between neurons

# fig, axs = plt.subplots(10, 10)
# window_length = 0

# for i in range(10):
#     for j in range(i,10):
#         axs[i,j].xcorr(Y[:,i+window_length], Y[:,j+window_length], maxlags=30)
#         axs[i,j].set_title('C' + str(i+1) + str(j+1))
#
# plt.xlabel('time shift')
# plt.tight_layout()

# Make a fig to plot the true and inferred network
plt.ion()
fig = plt.figure(3)
ax_true = fig.add_subplot(1, 2, 1, aspect="equal")
ax_test = fig.add_subplot(1, 2, 2, aspect="equal")

A = true_model.adjacency
W = true_model.weights
W_total = W.sum(1)
true_model.network.plot_LatentDistanceModel(A, W, ax=ax_true)

# Create a test model for fitting
test_model = LatentDistanceWeightsSparseBernoulliGLM(N, basis=basis, regression_kwargs=dict(rho=0.7, S_w=1, mu_b=-2))

test_model.add_data(Y)

# Fit with Gibbs sampling
def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0], m.network.L, m.network.lpf

def _update(m, itr):
    m.resample_model()
    ax_test.cla()
    test_model.network.plot_LatentDistanceModel(m.adjacency, m.weights, ax=ax_test, L_true=true_model.network.L)
    print("Iteration ", itr)
    print("LP:", m.log_likelihood())
    plt.pause(0.001)
    return _collect(m)

N_samples = 2000
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps, W_smpls, A_smpls, b_smpls, fr_smpls, L_smples, lpf_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
plt.figure(figsize=(4, 4))
plt.plot(lps)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.tight_layout()

# Plot the posterior mean and variance
W_mean = W_smpls[N_samples // 2:].mean(0)
A_mean = A_smpls[N_samples // 2:].mean(0)
fr_mean = fr_smpls[N_samples // 2:].mean(0)
fr_std = fr_smpls[N_samples // 2:].std(0)

plot_glm(Y, W_mean, A_mean, fr_mean, std_firingrates=3 * fr_std, title="Posterior Mean")

# plot true location and inferred location
from hips.plotting.colormaps import harvard_colors
from utils.utils import compute_optimal_rotation

color = harvard_colors()[0:10]
fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal")

N_select = 10
N_id_select = np.random.permutation(N)[0:N_select]
k = 0

for i in N_id_select:
    for j in range(N_samples):
        R = compute_optimal_rotation(L_smples[j], true_model.network.L)
        L_smples[j] = L_smples[j].dot(R)
        # affine translation
        D_t = np.mean(L_smples[j], 0) - np.mean(true_model.network.L, 0)
        L_smples[j] = L_smples[j] - D_t
    ax.scatter(L_smples[N_samples // 2:, i, 0], L_smples[N_samples // 2:, i, 1], s=20, c=color[k])
    k += 1

ax.scatter(true_model.network.L[N_id_select, 0], true_model.network.L[N_id_select, 1], s=300, c=color, edgecolor='0',
           lw=1, marker=(5, 1))

b = np.amax(abs(true_model.network.L)) + true_model.network.L[:].std() / 2.0

# Plot grids for origin
ax.plot([0, 0], [-b, b], ':k', lw=0.2)
ax.plot([-b, b], [0, 0], ':k', lw=0.2)

# Set the limits
ax.set_xlim([-b, b])
ax.set_ylim([-b, b])
ax.tick_params(axis='both', which='major', labelsize=16)

# Labels
ax.set_xlabel('Latent Dimension 1', fontsize=20)
ax.set_ylabel('Latent Dimension 2', fontsize=20)
plt.show()

plt.figure()
plt.plot(lpf_smpls[1:])

# Saving the objects:
#with open('TVpgGLM/results/sythetic_true_location1.pickle', 'wb') as f:
#   pickle.dump([lps, W, W_smpls, A_smpls, L_D, L_smples], f)


