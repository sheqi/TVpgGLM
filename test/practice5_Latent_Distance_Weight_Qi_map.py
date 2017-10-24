from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import numpy.random as npr
import tensorflow as tf
from pyglm.utils.utils import expand_scalar
from matplotlib import pyplot as plt
from edward.models import Normal

sess = ed.get_session()
# Simulated data
dim = 2
N = 30
r = 1
th = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L = np.hstack((x[:,None], y[:,None]))
W = np.zeros((N,N))
# Distance matrix
D = ((L[:, None, :] - L[None, :, :]) ** 2).sum(2)
sig = np.exp(-D / 2)
Sig = np.tile(sig[:,:,None,None], (1,1,1,1))

Mu = expand_scalar(0, (N, N, 1))

for n in range(N):
    for m in range(N):
        W[n, m] = npr.multivariate_normal(Mu[n, m], Sig[n, m])

# Initiaize the test model
L_estimate = Normal(loc=tf.zeros([N, dim]), scale=tf.ones([N, dim]))
xp = tf.tile(tf.reduce_sum(tf.pow(L_estimate, 2), 1, keep_dims=True), [1, N])
xp = xp + tf.transpose(xp) - 2 * tf.matmul(L_estimate, L_estimate, transpose_b=True)
xp = tf.exp(-xp / 2)
x  = Normal(loc=tf.zeros([N, N]), scale=xp)

# Inference using varitional inference
# qL_estimate = Normal(loc=tf.Variable(tf.random_normal([N,dim])),
#                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,dim]))))
#
# inference = ed.KLqp(latent_vars={L_estimate: qL_estimate}, data={x: W})
#
# inference.run(n_iter=100)
# L_estimate_samples = sess.run(qL_estimate)

# Inference using MAP
inference = ed.MAP([L_estimate], data={x: W})
inference.run(n_iter=1000)
mean = sess.run(L_estimate)

plt.scatter(mean[:,0], mean[:,1], color='blue')
plt.scatter(L[:,0], L[:,1], color='red')