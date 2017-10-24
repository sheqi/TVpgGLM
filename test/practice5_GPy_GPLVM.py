# GPLVM codes
import numpy as np
from matplotlib import pyplot as plt
import GPy # import GPy package
np.random.seed(12345)
GPy.plotting.change_plotting_library('plotly')

# Define dataset 
N = 100
k1 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,10,10,0.1,0.1]), ARD=True)
k2 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,0.1,10,0.1,10]), ARD=True)
k3 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[0.1,0.1,10,10,10]), ARD=True)
X = np.random.normal(0, 1, (N, 5))
A = np.random.multivariate_normal(np.zeros(N), k1.K(X), 1).T
B = np.random.multivariate_normal(np.zeros(N), k2.K(X), 1).T
C = np.random.multivariate_normal(np.zeros(N), k3.K(X), 1).T

Y = np.vstack((A,B,C))
labels = np.hstack((np.zeros(A.shape[0]), np.ones(B.shape[0]), np.ones(C.shape[0])*2))

input_dim = 2 # How many latent dimensions to use
kernel = GPy.kern.RBF(input_dim, 1, ARD=True) 

Q = input_dim
m_gplvm = GPy.models.GPLVM(Y, Q, kernel=GPy.kern.RBF(Q))
m_gplvm.kern.lengthscale = .2
m_gplvm.kern.variance = 1
m_gplvm.likelihood.variance = 1.

# Display info about gplvm
m_gplvm

# Optimization
m_gplvm.optimize(messages=1, max_iters=1e3)

# Plot
figure = GPy.plotting.plotting_library().figure(1, 2, 
                        shared_yaxes=True,
                        shared_xaxes=True,
                        subplot_titles=('Latent Space', 
                                        'Magnification',
                                        )
                            )

canvas = m_gplvm.plot_latent(labels=labels, figure=figure, col=(1), legend=False)
canvas = m_gplvm.plot_magnification(labels=labels, figure=figure, col=(2), legend=False)

GPy.plotting.show(canvas, filename='wishart_metric_notebook')