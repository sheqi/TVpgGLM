# Latent distance model for neural data
import numpy as np
import numpy.random as npr
from autograd import grad
from scipy.optimize import minimize
import autograd.numpy as atnp
from hips.inference.hmc import hmc
from pyglm.utils.utils import expand_scalar, compute_optimal_rotation
from matplotlib import pyplot as plt
"""
l_n ~ N(0, sigma^2 I)
W_{n', n} ~ N(0, exp(-||l_{n}-l_{n'}||_2^2/2)) for n' != n
"""

# Simulated data
dim = 2
N = 20
r = 1 + np.arange(N) // (N/2.)
th = np.linspace(0, 4 * np.pi, N, endpoint=False)
x = r * np.cos(th)
y = r * np.sin(th)
L = np.hstack((x[:, None], y[:, None]))
#w = 4
#s = 0.8
#x = s * (np.arange(N) % w)
#y = s * (np.arange(N) // w)
#L = np.hstack((x[:,None], y[:,None]))
W = np.zeros((N, N))
# Distance matrix
D = ((L[:, None, :] - L[None, :, :]) ** 2).sum(2)
sig = np.exp(-D / 2)
Sig = np.tile(sig[:, :, None, None], (1, 1, 1, 1))

# Covariance of prior on l_{n}
sigma = 1

Mu = expand_scalar(0, (N, N, 1))
L_estimate = np.sqrt(sigma) * np.random.randn(N, dim)

for n in range(N):
    for m in range(N):
        W[n, m] = npr.multivariate_normal(Mu[n, m], Sig[n, m])


def _hmc_log_probability(N, dim, L, W, sigma):
    """
    Compute the log probability as a function of L.
    This allows us to take the gradients wrt L using autograd.
    :param L:
    :return:
    """

    # Compute pairwise distance
    L1 = atnp.reshape(L, (N, 1, dim))
    L2 = atnp.reshape(L, (1, N, dim))

    X = W
    # Get the covariance and precision
    Sig = atnp.exp((-atnp.sum((L1 - L2) ** 2, axis=2)) / 2)
    Lmb = 1. / Sig

    lp = atnp.sum(-0.5 * X ** 2 * Lmb)

    # Log prior of L under spherical Gaussian prior
    lp += -0.5 * atnp.sum(L * L / sigma)

    return lp


def _resample_sigma(L):
    """
    Resample sigma under an inverse gamma prior, sigma ~ IG(1,1)
    :return:
    """

    a_prior = 1.0
    b_prior = 1.0

    a_post = a_prior + L.size / 2.0
    b_post = b_prior + (L ** 2).sum() / 2.0

    from scipy.stats import invgamma
    sigma = invgamma.rvs(a=a_post, scale=b_post)

    return sigma

def plot_LatentDistanceModel(W, L, N, L_true=None, ax=None):
    """
    If D==2, plot the embedded nodes and the connections between them

    :param L_true:  If given, rotate the inferred features to match F_true
    :return:
    """
    # Color the weights by the
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdBu")
    W_lim = abs(W[:,:]).max()
    W_rel = (W[:,:] - (-W_lim)) / (2*W_lim)

    if ax is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111, aspect="equal")

    # If true locations are given, rotate L to match L_true
    if L_true is not None:
        R = compute_optimal_rotation(L, L_true)
        L = L.dot(R)

    # Scatter plot the node embeddings
    # Plot the edges between nodes
    for n1 in range(N):
        for n2 in range(N):
            ax.plot([L[n1,0], L[n2,0]],
                    [L[n1,1], L[n2,1]],
                    '-', color=cmap(W_rel[n1,n2]),
                    lw=1.0)
    ax.plot(L[:,0], L[:,1], 's', color='k', markerfacecolor='k', markeredgecolor='k')

    # Get extreme feature values
    b = np.amax(abs(L)) + L[:].std() / 2.0

    # Plot grids for origin
    ax.plot([0,0], [-b,b], ':k', lw=0.5)
    ax.plot([-b,b], [0,0], ':k', lw=0.5)

    # Set the limits
    ax.set_xlim([-b,b])
    ax.set_ylim([-b,b])

    # Labels
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    plt.show()

    return ax

# Inference using bfgs method

lp = lambda Lflat: -_hmc_log_probability(N, dim, atnp.reshape(Lflat, (N, 2)), W, sigma)
dlp = grad(lp)

for i in range(1000):
    res = minimize(lp, np.ravel(L_estimate), jac=dlp, method="bfgs")
    L_estimate = np.reshape(res.x, (N, 2))

# Debug here, because the two directed weights are ploted together
# With different strength
plot_LatentDistanceModel(W, L_estimate, N, L_true=L)
plot_LatentDistanceModel(W, L, N)
