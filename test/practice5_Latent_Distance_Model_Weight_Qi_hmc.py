# Latent distance model for neural data
import numpy as np
import numpy.random as npr
from autograd import grad
from hips.inference.hmc import hmc
from pybasicbayes.distributions import Gaussian
from pyglm.utils.utils import expand_scalar, compute_optimal_rotation
from matplotlib import pyplot as plt
import seaborn as sns
np.random.seed(20)
"""
l_n ~ N(0, sigma^2 I)
W_{n', n} ~ N(0, exp(-||l_{n}-l_{n'}||_2^2/2)) for n' != n
"""

# Simulated data
dim = 2
N = 5
#r = 2 + np.arange(N) // (N/2.)
r = 2
th = np.linspace(0, 2 * np.pi, N, endpoint=False)
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
sig = 5*np.exp(-D/2)
Sig = np.tile(sig[:, :, None, None], (1, 1, 1, 1))

# Covariance of prior on l_{n}
sigma = 2

Mu = expand_scalar(0, (N, N, 1))
L_estimate = np.sqrt(sigma) * np.random.randn(N, dim)
# L_estimate = L

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
    import autograd.numpy as atnp

    # Compute pairwise distance
    L1 = atnp.reshape(L, (N, 1, dim))
    L2 = atnp.reshape(L, (1, N, dim))

    X = W
    # Get the covariance and precision
    Sig1 = 5*atnp.exp(-atnp.sum((L1 - L2) ** 2, axis=2)/2) + 1e-4
    # Sig1 = atnp.sum((L1 - L2) ** 2, axis=2)
    Lmb = 1. / Sig1

    lp = -0.5 * atnp.sum(atnp.log(2 * np.pi * Sig1)) + atnp.sum(-0.5 * X ** 2 * Lmb)

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

# Inference using HMC method
N_samples = 500
smpls = np.zeros((N_samples,N,dim))
smpls[0] = L_estimate

lp1 = np.zeros(N_samples)
#lp1[0] = lp(L_estimate)
a = np.zeros(N_samples)
W_all = np.zeros((N_samples,N,N))

for s in np.arange(1,N_samples):

    W1 = W + np.random.normal(0.1,0.1)

    lp = lambda L1: _hmc_log_probability(N, dim, L1, W, sigma)
    dlp = grad(lp)
    stepsz = 0.005
    nsteps = 10
    accept_rate = 0.9
    smpls[s], stepsz, accept_rate= \
        hmc(lp, dlp, stepsz, nsteps, smpls[s-1], negative_log_prob=False, avg_accept_rate=accept_rate,
                adaptive_step_sz=True)

    lp1[s] = lp(smpls[s])
    sigma = _resample_sigma(smpls[s])
    a[s] = sigma
    W_all[s-1] = W1
    print(sigma)

for s in range(N_samples):
    R = compute_optimal_rotation(smpls[s], L)
    smpls[s] = np.dot(smpls[s], R)

L_estimate = smpls[N_samples // 2:].mean(0)

# Debug here, because the two directed weights are ploted together
# With different strength

#plot_LatentDistanceModel(W, L_estimate, N, L_true=L)
#plot_LatentDistanceModel(W, L, N)
plt.figure(1)
plt.scatter(smpls[-100:,:,0],smpls[-100:,:,1])
plt.scatter(L[:,0], L[:,1],color='r')
plt.figure(2)
plt.plot(lp1)
plt.figure(3)
plt.plot(W.reshape(N*N))
sns.tsplot(W_all.reshape(N_samples,N*N))
