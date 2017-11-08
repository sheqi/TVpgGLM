# experimental data for tv analysis

# hyperparameter for variance of weights trajectory:
# 0.0015

import sys

sys.path.append("/Users/roger/Dropbox/pyglm-master")
sys.path.append("/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/libs")

import numpy as np
np.random.seed(1)

import pickle
import seaborn as sns
sns.set_style("white")

from pybasicbayes.util.text import progprint_xrange
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

# Neuron
N1 = 154
N2 = 272

##############################
##static model on 1st half##
##############################

T = 3000
B = 1
N = 2
L = 10
Y_1st = np.vstack((Y0[:,N1],Y0[:,N2]))
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

# Posterior mean and variance
W_mean1 = W_smpls[N_samples//2:].mean(0)
W_std1 = W_smpls[N_samples//2:].std(0)
A_mean1 = A_smpls[N_samples//2:].mean(0)
fr_mean1 = fr_smpls[N_samples//2:].mean(0)
fr_std1 = fr_smpls[N_samples//2:].std(0)


###############################
##static model on 2nd half##
###############################
Y_2nd = np.vstack((Y1_0[:,N1],Y1_0[:,N2]))
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


# Plot the posterior mean and variance
W_mean2 = W_smpls[N_samples//2:].mean(0)
W_std2 = W_smpls[N_samples//2:].std(0)
A_mean2 = A_smpls[N_samples//2:].mean(0)
fr_mean2 = fr_smpls[N_samples//2:].mean(0)
fr_std2 = fr_smpls[N_samples//2:].std(0)

###############################
##static model on 3rd half##
###############################
Y_3rd = np.vstack((Y2[:,N1],Y2[:,N2]))
Y_3rd = Y_3rd.transpose()

basis = cosine_basis(B=B, L=L) / L
# Make a test regression and fit it
test_model = BernoulliGLM(N, basis=basis,
                       regression_kwargs=dict(rho=1, S_w=10, mu_b=-2.))

test_model.add_data(Y_3rd)

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
lps3, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))


# Plot the posterior mean and variance
W_mean3 = W_smpls[N_samples//2:].mean(0)
W_std3 = W_smpls[N_samples//2:].std(0)
A_mean3 = A_smpls[N_samples//2:].mean(0)
fr_mean3 = fr_smpls[N_samples//2:].mean(0)
fr_std3 = fr_smpls[N_samples//2:].std(0)

##########################################################
##Time-varying model analysis training the whole process##
##########################################################
# Create a test model for fitting
Y_123 = np.vstack((Y0,Y1_0,Y2))
Y_123 = np.vstack((Y_123[:,N1],Y_123[:,N2]))
Y_123 = Y_123.transpose()

T  = 9000

N_samples = 50

test_model = \
    SparseBernoulliGLM_f(T, N, B, basis=basis,
                         regression_kwargs=dict(rho=1, mu_w=0,
                                                S_w=0.01, mu_b=-2, S_b=0.0001))
test_model.add_data(Y_123)

def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]


def _update(m, itr):
    m.resample_model()
    return _collect(m)

samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps4, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the posterior mean and variance
W_mean4 = W_smpls[N_samples // 2:].mean(0)
W_std4 = W_smpls[N_samples//2 :].std(0)
A_mean4 = A_smpls[N_samples // 2:].mean(0)
fr_mean4 = fr_smpls[N_samples // 2:].mean(0)
fr_std4 = fr_smpls[N_samples // 2:].std(0)

##########################################################
##Static model analysis training the whole process##
##########################################################
T  = 9000

N_samples = 500

test_model = \
    test_model = BernoulliGLM(N, basis=basis,
                              regression_kwargs=dict(rho=1, S_w=10, mu_b=-2.))
test_model.add_data(Y_123)

def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]


def _update(m, itr):
    m.resample_model()
    return _collect(m)

samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps5, W_smpls1, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the posterior mean and variance
W_mean5 = W_smpls1[N_samples // 2:].mean(0)
W_std5 = W_smpls1[N_samples//2 :].std(0)
A_mean5 = A_smpls[N_samples // 2:].mean(0)
fr_mean5 = fr_smpls[N_samples // 2:].mean(0)
fr_std5 = fr_smpls[N_samples // 2:].std(0)

# Saving the objects:
with open('TVpgGLM/results/exp_tv_N2.pickle', 'wb') as f:
    pickle.dump([lps1, lps2, lps3, lps4, lps5,
                 W_mean1, W_mean2, W_mean3, W_mean4, W_mean5, W_std1, W_std2, W_std3, W_std4, W_std5, W_smpls,
                 Y_1st, Y_2nd, Y_3rd, Y_123,
                 fr_mean1, fr_mean2, fr_mean3, fr_mean4, fr_mean5, fr_std1, fr_std2, fr_std3, fr_std4, fr_mean5
                ],f)

