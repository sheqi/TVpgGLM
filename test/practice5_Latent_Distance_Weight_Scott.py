# Latent distance model for neural data
import numpy as np
import numpy.random as npr
from autograd import grad
from hips.inference.hmc import hmc
from pybasicbayes.distributions import Gaussian
from pyglm.utils.utils import expand_scalar, compute_optimal_rotation
from matplotlib import pyplot as plt

# Simulated data
dim = 2
N = 20
r = 1
th = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L = np.hstack((x[:,None], y[:,None]))
#w = 4
#s = 0.8
#x = s * (np.arange(N) % w)
#y = s * (np.arange(N) // w)
#L = np.hstack((x[:,None], y[:,None]))


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

# Inference
def _hmc_log_probability(N, dim, L, W):
    """
    Compute the log probability as a function of L.
    This allows us to take the gradients wrt L using autograd.
    :param L:
    :param A:
    :return:
    """
    import autograd.numpy as anp

    # Compute pairwise distance
    L1 = anp.reshape(L, (N, 1, dim))
    L2 = anp.reshape(L, (1, N, dim))
    # Mu = a * anp.sqrt(anp.sum((L1-L2)**2, axis=2)) + b
    Mu = -anp.sum((L1 - L2) ** 2, axis=2)

    X = (W - Mu[:, :, None])

    # Get the covariance and precision
    Sig = 0.01
    Lmb = 1. / Sig

    lp = anp.sum(-0.5 * X ** 2 * Lmb)

    # Log prior of L under spherical Gaussian prior
    lp += -0.5 * anp.sum(L * L)

    return lp


def plot_LatentDistanceModel(W, L, N, L_true=None, ax=None):
    """
    If D==2, plot the embedded nodes and the connections between them

    :param L_true:  If given, rotate the inferred features to match F_true
    :return:
    """
    # Color the weights by the
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdBu")
    W_lim = abs(W[:, :]).max()
    W_rel = (W[:, :] - (-W_lim)) / (2 * W_lim)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")

    # If true locations are given, rotate L to match L_true
    if L_true is not None:
        R = compute_optimal_rotation(L, L_true)
        L = L.dot(R)

    # Scatter plot the node embeddings
    # Plot the edges between nodes
    for n1 in range(N):
        for n2 in range(N):
            ax.plot([L[n1, 0], L[n2, 0]], [L[n1, 1], L[n2, 1]], '-', color=cmap(W_rel[n1, n2]), lw=1.0)
    ax.plot(L[:, 0], L[:, 1], 's', color='k', markerfacecolor='k', markeredgecolor='k')

    # Get extreme feature values
    b = np.amax(abs(L)) + L[:].std() / 2.0

    # Plot grids for origin
    ax.plot([0, 0], [-b, b], ':k', lw=0.5)
    ax.plot([-b, b], [0, 0], ':k', lw=0.5)

    # Set the limits
    ax.set_xlim([-b, b])
    ax.set_ylim([-b, b])

    # Labels
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    plt.show()

    return ax

for i in range(1000):

    L1 = L_estimate
    lp = lambda L1: _hmc_log_probability(N, dim, L1, W)
    dlp = grad(lp)

    stepsz = 0.001
    nsteps = 10
    L_estimate = hmc(lp, dlp, stepsz, nsteps, L1.copy(), negative_log_prob=False)


D1 = ((L_estimate[:, None, :] - L_estimate[None, :, :]) ** 2).sum(2)
W_estimate = -D1

plot_LatentDistanceModel(W_estimate, L_estimate, N)
plot_LatentDistanceModel(W, L, N)


