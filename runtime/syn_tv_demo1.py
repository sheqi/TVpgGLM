#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:12:47 2017

@author: roger
"""

# hyperparameter variance: 0.015
import sys

sys.path.append("/Users/roger/Dropbox/pyglm-master")
sys.path.append("/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/libs")

import numpy as np

np.random.seed(100)

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_style("white")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10,
            'font.size': 15, 'axes.labelsize':15,
            'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

from pybasicbayes.util.text import progprint_xrange
from pyglm.utils.basis import cosine_basis
from pyglm.plotting import plot_glm
from models_tv import SparseBernoulliGLM_f

T = 1000  # Number of time bins to generate
N = 10  # Number of neurons
B = 1  # Number of "basis functions"
L = 10  # Autoregressive window of influence

# Create a cosine basis to model smooth influence of
# spikes on one neuron on the later spikes of others.
basis = cosine_basis(B=B, L=L, a=1.0 / 10) / L

# Generate some data from a model with self inhibition
# The model structure has the info for the network and regression we used
true_model = \
    SparseBernoulliGLM_f(T, N, B, basis=basis,
                            regression_kwargs=dict(rho=1, mu_w=0,
                                                   S_w=0.001, mu_b=-2, S_b=0.0001))

_, Y = true_model.generate(T=T, keep=True)

# Plot the true model
fig, axs, handles = true_model.plot()
plt.figure()
sns.heatmap(np.transpose(Y), xticklabels=False)

# Create a test model for fitting
test_model = \
    SparseBernoulliGLM_f(T, N, B, basis=basis,
                         regression_kwargs=dict(rho=1, mu_w=0, S_w=0.001, mu_b=-2, S_b=0.0001))

test_model.add_data(Y)

def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]

def _update(m, itr):
    m.resample_model()
    return _collect(m)

N_samples = 100
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the posterior mean and variance
W_mean = W_smpls[N_samples // 2:].mean(0)
A_mean = A_smpls[N_samples // 2:].mean(0)
fr_mean = fr_smpls[N_samples // 2:].mean(0)
fr_std = fr_smpls[N_samples // 2:].std(0)

fig, _, _ = plot_glm(Y, W_mean[:, 0, :, :], A_mean, fr_mean,
                    std_firingrates=3 * fr_std, title="Posterior Mean")

# Saving the objects:
#with open('TVpgGLM/results/sythetic_tv_N10.pickle', 'wb') as f:
#    pickle.dump([true_model.means[0], true_model.weights, fr_mean, fr_std, W_smpls, Y], f)