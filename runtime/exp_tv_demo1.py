# experimental data for tv analysis
import sys

sys.path.append("/Users/roger/Dropbox/pyglm-master")
sys.path.append("/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/libs")

import numpy as np
np.random.seed(1)

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from pybasicbayes.util.text import progprint_xrange
from pyglm.plotting import plot_glm
from pyglm.utils.basis import cosine_basis
from pyglm.models import BernoulliGLM
from models_tv import SparseBernoulliGLM_f

# load experimental data from hippocampus
import scipy.io as sio
matfn = '/Users/roger/Dropbox/pyglm-master/Achilles_10252013/results3.mat'
data = sio.loadmat(matfn)

Y0 = data['pre_spk_1']
Y1_0 = data['dur_spk_0']
Y1_1 = data['dur_spk_1']
Y2 = data['aft_spk_1']


##############################
##static model on first half##
##############################
T = 3000
B = 1
N = 2
L = 10
Y_1st = np.vstack((Y0[:,273],Y0[:,272]))
Y_1st = Y_1st.transpose()

basis = cosine_basis(B=B, L=L) / L
# Make a test regression and fit it
test_model = BernoulliGLM(N, basis=basis,
                       regression_kwargs=dict(rho=1, S_w=10, mu_b=-2.))

test_model.add_data(Y_1st)

# Fit with Gibbs sampling
def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]

def _update(m, itr):
    m.resample_model()
    return _collect(m)

N_samples = 500
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps1, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
fig = plt.figure(figsize=(4,4))
plt.plot(lps1)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.tight_layout()

# Plot the posterior mean and variance
W_mean1 = W_smpls[N_samples//2:].mean(0)
W_std1 = W_smpls[N_samples//2:].std(0)
A_mean1 = A_smpls[N_samples//2:].mean(0)
fr_mean1 = fr_smpls[N_samples//2:].mean(0)
fr_std1 = fr_smpls[N_samples//2:].std(0)

fig, _, _ = plot_glm(Y_1st, W_mean1, A_mean1, fr_mean1,
                     std_firingrates=3*fr_std1, title="Posterior Mean")

plt.figure()
sns.heatmap(W_mean1[:,:,0])


###############################
##static model on second half##
###############################
Y_2nd = np.vstack((Y1_0[:,273],Y1_0[:,272]))
Y_2nd = Y_2nd.transpose()

basis = cosine_basis(B=B, L=L) / L
# Make a test regression and fit it
test_model = BernoulliGLM(N, basis=basis,
                       regression_kwargs=dict(rho=1, S_w=10, mu_b=-2.))

test_model.add_data(Y_2nd)

# Fit with Gibbs sampling
def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]

def _update(m, itr):
    m.resample_model()
    return _collect(m)

N_samples = 500
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps2, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
fig = plt.figure(figsize=(4,4))
plt.plot(lps2)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.tight_layout()

# Plot the posterior mean and variance
W_mean2 = W_smpls[N_samples//2:].mean(0)
W_std2 = W_smpls[N_samples//2:].std(0)
A_mean2 = A_smpls[N_samples//2:].mean(0)
fr_mean2 = fr_smpls[N_samples//2:].mean(0)
fr_std2 = fr_smpls[N_samples//2:].std(0)

fig, _, _ = plot_glm(Y_2nd, W_mean2, A_mean2, fr_mean2,
                     std_firingrates=3*fr_std2, title="Posterior Mean")

plt.figure()
sns.heatmap(W_mean2[:,:,0])

###############################
##Time-varying model analysis##
###############################
# Create a test model for fitting
Y_12 = np.vstack((Y0,Y1_0))
Y_12 = np.vstack((Y_12[:,273],Y_12[:,272]))
Y_12 = Y_12.transpose()

T  = 6000

N_samples = 5

test_model = \
    SparseBernoulliGLM_f(T, N, B, basis=basis,
                         regression_kwargs=dict(rho=1, mu_w=0,
                                                S_w=0.01, mu_b=-2, S_b=0.0001))
test_model.add_data(Y_12)

def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]


def _update(m, itr):
    m.resample_model()
    #    test_model.plot(handles=handles,
    #                    pltslice=slice(0, 500),
    #                    title="Sample {}".format(itr+1))
    return _collect(m)

samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps3, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
fig = plt.figure(figsize=(4, 4))
plt.plot(lps3)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.tight_layout()

# Plot the posterior mean and variance
W_mean3 = W_smpls[N_samples // 2:].mean(0)
W_std3 = W_smpls[N_samples//2 :].std(0)
A_mean3 = A_smpls[N_samples // 2:].mean(0)
fr_mean3 = fr_smpls[N_samples // 2:].mean(0)
fr_std3 = fr_smpls[N_samples // 2:].std(0)

fig, _, _ = plot_glm(Y_1st, W_mean3[:, 0, :, :], A_mean3, fr_mean3,
                     std_firingrates=3 * fr_std3, title="Posterior Mean")

# Plot weights comparison
from hips.plotting.colormaps import harvard_colors
color = harvard_colors()[0:10]
# Plot weights comparison
fig, axs = plt.subplots(N, N)
for i in range(N):
    for j in range(N):
        sns.tsplot(data=W_smpls[N_samples // 2:,i,0:T,j, 0], ax=axs[i,j], color=color[1], condition='estimated')
        axs[i,j].set_xlabel('Time', fontweight="bold")
        axs[i,j].set_ylabel('Weights', fontweight="bold")
        axs[i,j].legend(loc="upper center", ncol=2, prop={'size':15})

# Saving the objects:
with open('TVpgGLM/results/exp_tv_N2.pickle', 'wb') as f:
    pickle.dump([lps1, lps2, lps3,
                 W_mean1, W_mean2, W_mean3, W_std1, W_std2, W_std3,
                 Y_1st, Y_2nd, Y_12,
                 fr_mean1, fr_mean2, fr_mean3, fr_std1, fr_std2, fr_std3
                 ],f)

