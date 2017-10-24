#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:12:47 2017

@author: roger
"""
import sys
sys.path.append("/Users/pillowlab/Dropbox/pyglm-master")

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt
import seaborn as sns
import xcorr

from pybasicbayes.util.text import progprint_xrange
from pyglm.utils.basis import cosine_basis
from pyglm.plotting import plot_glm
from models_tv import SparseBernoulliGLM_f
from hips.plotting.colormaps import harvard_colors

sns.set_style("white")
paper_rc = {'lines.linewidth': 2.5, 'lines.markersize': 10, 'font.size': 15,
            'axes.labelsize':15, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

T = 2000  # Number of time bins to generate
N = 2  # Number of neurons
B = 1  # Number of "basis functions"
L = 20  # Autoregressive window of influence

basis = cosine_basis(B=B, L=L, a=1.0 / 20) / L

# Generate some data from a model with self inhibition
# The model structure has the info for the network and regression we used
true_model = \
    SparseBernoulliGLM_f(T, N, B, basis=basis, regression_kwargs=dict(rho=1, mu_w=0, S_w=0.001, mu_b=-2, S_b=0.0001))

# Network randomly assigned weights
# sine wave
Fs = [1000, 2000, 3000]
f = 5
sample = T
x = np.arange(sample)

# Randomly assiged network weights
true_model.regressions[0].W[:, 0, 0] = 2 - 2 * np.exp(-0.0005 * x)
true_model.regressions[1].W[:, 1, 0] = 2 - 2 * np.exp(-0.0005 * x)
true_model.regressions[0].W[:, 1, 0] = -2 + 2 * np.exp(-0.0005 * x)
true_model.regressions[1].W[:, 0, 0] = 1.0 * np.cos(2 * np.pi * f * x / Fs[2] + np.pi) + 1.0

_, Y = true_model.generate(T=T, keep=True)

# Plot the true model
fig, axs, handles = true_model.plot()
plt.figure()
sns.heatmap(np.transpose(Y), xticklabels=False)

# Create a test model for fitting
test_model = \
    SparseBernoulliGLM_f(T, N, B, basis=basis, regression_kwargs=dict(rho=1, mu_w=0, S_w=0.001, mu_b=-2, S_b=0.0001))

test_model.add_data(Y)


def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]


def _update(m, itr):
    m.resample_model()
    return _collect(m)

N_samples = 50
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
fig = plt.figure(figsize=(4, 4))
plt.plot(lps)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.tight_layout()

# Plot the posterior mean and variance
W_mean = W_smpls[N_samples // 2:].mean(0)
A_mean = A_smpls[N_samples // 2:].mean(0)
fr_mean = fr_smpls[N_samples // 2:].mean(0)
fr_std = fr_smpls[N_samples // 2:].std(0)

fig, _, _ = plot_glm(Y, W_mean[:, 0, :, :], A_mean, fr_mean, std_firingrates=3 * fr_std, title="Posterior Mean")

color = harvard_colors()[0:10]
# Plot weights comparison
fig, axs = plt.subplots(N, N)
for i in range(N):
    for j in range(N):
        sns.tsplot(data=true_model.regressions[i].W[0:1000, j, 0], ax=axs[i, j], color=color[0], condition='true')
        sns.tsplot(data=W_smpls[N_samples // 2:,i,0:1000,j, 0],ax=axs[i,j],color=color[1],condition='estimated')
        axs[i,j].set_xlabel('Time', fontweight="bold")
        axs[i,j].set_ylabel('Weights',fontweight="bold")
        axs[i,j].legend(loc="upper center", ncol=2, prop={'size':15})

# Plot cross-correlation
fig1, axs = plt.subplots(N, N, sharey=True)

fr_true = xcorr.xcorr(true_model.means[0], dtmax=20)
fr_est = xcorr.xcorr(fr_mean, dtmax=20)

for i0 in range(N):
    for j0 in range(N):
        axs[i0, j0].plot(fr_true[i0, j0, :])
        axs[i0, j0].plot(fr_est[i0, j0, :])

for i0 in range(N):
    for j0 in range(N):
        axs[i0, j0].set_xticklabels([])
        axs[i0, j0].set_yticklabels([])
