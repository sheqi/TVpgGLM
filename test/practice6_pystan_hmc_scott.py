# HMC estimation of the new model
# Simulated data
import numpy as np
import numpy.random as npr
from pyglm.utils.utils import expand_scalar, compute_optimal_rotation
from matplotlib import pyplot as plt
import seaborn as sns

dim = 2
N = 20
r = 1 + np.arange(N) // (N/2.)
th = np.linspace(0, 4 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L = np.hstack((x[:, None], y[:, None]))

W = np.zeros((N,N))

# Distance matrix
D = ((L[:, None, :] - L[None, :, :]) ** 2).sum(2)
Mu = -D
Mu = np.tile(Mu[:,:,None], (1,1,1))
sig = 0.01*np.eye(N)
Sig = np.tile(sig[:,:,None,None], (1,1,1,1))

L_estimate = np.random.randn(N, dim)

for n in range(N):
    for m in range(N):
        W[n, m] = npr.multivariate_normal(Mu[n, m], Sig[n, m])


model = """
data {
   int<lower=1> N; // Number of data points
   matrix[N,N] W;      // the 1st predictor
}
parameters {
   matrix[N,2] l;
   real<lower=0> sigma;
   real<lower=0> eta;
   real b;
}
model {
   sigma ~ inv_gamma(1,1);
   for (i in 1:N)
        for (j in 1:2)
            l[i,j] ~ normal(0, sigma);
   for (i in 1:N)
        for (j in 1:N)
            W[i,j] ~ normal(-squared_distance(l[i,:],l[j,:])+b, eta); 
}
generated quantities {
    matrix[N,N] W_pred;
    for (i in 1:N)
        for (j in 1:N)
            W_pred[i,j] = normal_rng(-squared_distance(l[i,:],l[j,:])+b, eta);
}
"""

data = {'N':N, 'W':W}

import pystan
fit = pystan.stan(model_code=model, data=data, iter=500, chains=4, control=dict(stepsize=0.1))

print(fit)

samples = fit.extract(permuted=True)
L_estimate_all = samples['l']

#The results need to rotate
for i in range(1000):
    R = compute_optimal_rotation(L_estimate_all[i,:,:], L)
    L_estimate_all[i, :, :] = np.dot(L_estimate_all[i, :, :],R)

W_pred = samples['W_pred']
L_estimate = np.mean(L_estimate_all,0)
plt.subplot(321)
plt.scatter(L[:,0],L[:,1])
plt.subplot(322)
plt.scatter(L_estimate[:,0], L_estimate[:,1])
plt.subplot(323)
sns.heatmap(W)
plt.subplot(324)
sns.heatmap(np.mean(W_pred,0))
plt.subplot(325)
for i in range(N):
    sns.kdeplot(L_estimate_all[:,i,0])
plt.subplot(326)
for j in range(N):
    sns.kdeplot(L_estimate_all[:,j,1])



