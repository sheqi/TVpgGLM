#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:48:21 2017
Time-varying weights network
@author: roger
"""
import abc
import numpy as np

from pybasicbayes.abstractions import GibbsSampling
from pybasicbayes.distributions import Gaussian, GaussianFixedMean, GaussianFixedCov

from pyglm.utils.utils import expand_scalar, expand_cov

class _NetworkModel(GibbsSampling):
    def __init__(self, T, N , B, **kwargs):
        """
        Only extra requirement is that we explicitly tell it the
        number of nodes and the dimensionality of the weights in the constructor.

        :param N: Number of nodes
        :param T: Number of time bins
        :param B: Dimensionality of the weights
        :param N0: N*T
        """
        self.T, self.N, self.B = T, N, B

    @abc.abstractmethod
    def resample(self,data=[]):
        """
        Every network mixin's resample method should call its parent in its
        first line. That way we can ensure that this base method is called
        first, and that each mixin is resampled appropriately.

        :param data: an adjacency matrix and a weight matrix
            A, W = data
            A in [0,1]^{N x N} where rows are incoming and columns are outgoing nodes
            W in [0,1]^{NT x NT x B} where rows are incoming and columns are outgoing nodes
        """
        assert isinstance(data, tuple)
        A, W = data
        T, N, B = self.T, self.N, self.B
        assert A.shape == (N, N)
        assert A.dtype == bool
        assert W.shape == (N, T, N, B)

    @abc.abstractproperty
    @property
    def mu_W(self):
        """
        NxNxB array of mean weights
        """
        raise NotImplementedError

    @abc.abstractproperty
    @property
    def sigma_W(self):
        """
        NxNxBxB array with conditional covariances of each weight
        """
        raise NotImplementedError

    @abc.abstractproperty
    @property
    def rho(self):
        """
        Connection probability
        :return: NxN matrix with values in [0,1]
        """
        pass

    ## TODO: Add properties for info form weight parameters
    def log_likelihood(self, x):
        # TODO
        return 0

    def rvs(self,size=[]):
        # TODO
        return None

### Time-varying weight model
class _FixedWeightsMixin(_NetworkModel):
    def __init__(self, T, N, B,
                 mu=0.0, sigma=0.015,
                 mu_self=None, sigma_self=None,
                 **kwargs):
        super(_FixedWeightsMixin, self).__init__(T, N, B)
        self._mu = expand_scalar(mu, (N, T, N, B))
        self._sigma = expand_cov(sigma, (N, T, N, B, B))

        # initialize sigma
        self._sigma[0, :, 0, :, :] = 0.015
        self._sigma[1, :, 1, :, :] = 0.015
        self._sigma[1, :, 0, :, :] = 0.015
        self._sigma[0, :, 1, :, :] = 0.015

        if (mu_self is not None) and (sigma_self is not None):
            self._mu[np.arange(N), :, np.arange(N), :] = expand_scalar(mu_self, (N, T, B))
            self._sigma[np.arange(N), :, np.arange(N), :] = expand_cov(sigma_self, (N, T, B, B))

    @property
    # property: read only
    def mu_W(self):
        return self._mu

    @property
    def sigma_W(self):
        return self._sigma

    def resample(self,data=[]):
        super(_FixedWeightsMixin, self).resample(data)
        A, W = data

        # resample sigma in gibbs sampling
        # self._resample_sigma(W)

    def _resample_sigma(self,W):
        """
        Resample sigma under an inverse gamma prior, sigma ~ IG(1,1)
        :return:
        """
        T = self.T

        a_prior = 1.0
        b_prior = 1.0

        a_post = a_prior + T / 2.0
        b_post = b_prior + (W[:,1:T,:,:]-W[:,0:T-1,:,:]).sum(1) / 2.0

        from scipy.stats import invgamma
        for i in range(self.N):
            for j in range(self.N):
                self._sigma[i,:,j,:,:] = invgamma.rvs(a=a_post, scale=b_post[i,j,0])


class _IndependentGaussianMixin(_NetworkModel):
    """
    Each weight is an independent Gaussian with a shared NIW prior.
    Special case the self-connections.
    """
    def __init__(self, N, B,
                 mu_0=0.0, sigma_0=1.0, kappa_0=1.0, nu_0=3.0,
                 is_diagonal_weight_special=True,
                 **kwargs):
        super(_IndependentGaussianMixin, self).__init__(N, B)

        mu_0 = expand_scalar(mu_0, (B,))
        sigma_0 = expand_cov(sigma_0, (B,B))
        self._gaussian = Gaussian(mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, nu_0=max(nu_0, B+2.))

        self.is_diagonal_weight_special = is_diagonal_weight_special
        if is_diagonal_weight_special:
            self._self_gaussian = \
                Gaussian(mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, nu_0=nu_0)

    @property
    def mu_W(self):
        N, B = self.N, self.B
        mu = np.zeros((N, N, B))
        if self.is_diagonal_weight_special:
            # Set off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            mu[mask] = self._gaussian.mu

            # set diagonal weights
            mask = np.eye(N).astype(bool)
            mu[mask] = self._self_gaussian.mu

        else:
            mu = np.tile(self._gaussian.mu[None,None,:], (N, N, 1))
        return mu

    @property
    def sigma_W(self):
        N, B = self.N, self.B
        if self.is_diagonal_weight_special:
            sigma = np.zeros((N, N, B, B))
            # Set off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            sigma[mask] = self._gaussian.sigma

            # set diagonal weights
            mask = np.eye(N).astype(bool)
            sigma[mask] = self._self_gaussian.sigma

        else:
            sigma = np.tile(self._gaussian.mu[None, None, :, :], (N, N, 1, 1))
        return sigma

    def resample(self, data=[]):
        super(_IndependentGaussianMixin, self).resample(data)
        A, W = data
        N, B = self.N, self.B
        if self.is_diagonal_weight_special:
            # Resample prior for off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            mask = mask & A
            self._gaussian.resample(W[mask])

            # Resample prior for diagonal weights
            mask = np.eye(N).astype(bool) & A
            self._self_gaussian.resample(W[mask])

        else:
            # Resample prior for all weights
            self._gaussian.resample(W[A])


class _LatentDistanceModelGaussianMixin(_NetworkModel):
    """
    l_n ~ N(0, sigma^2 I)
    W_{n', n} ~ N(A * ||l_{n'} - l_{n}||_2^2 + b, ) for n' != n
    """
    def __init__(self, N, B=1, dim=2,
                 b=0.5,
                 sigma=None, Sigma_0=None, nu_0=None,
                 mu_self=0.0, eta=0.01):

        super(_LatentDistanceModelGaussianMixin, self).__init__(N,B)
        self.B = B
        self.dim = dim

        self.b = b
        self.eta = eta
        self.L = np.sqrt(eta) * np.random.randn(N,dim)

        if Sigma_0 is None:
            Sigma_0 = np.eye(B)

        if nu_0 is None:
            nu_0 = B + 2

        self.cov = GaussianFixedMean(mu=np.zeros(B), sigma=sigma, lmbda_0=Sigma_0, nu_0=nu_0)

        # Special case self-weights (along the diagonal)
        self._self_gaussian = Gaussian(mu_0=mu_self*np.ones(B),
                                       sigma_0=Sigma_0,
                                       nu_0=nu_0,
                                       kappa_0=1.0)

    @property
    def D(self):
        # return np.sqrt(((self.L[:, None, :] - self.L[None, :, :]) ** 2).sum(2))
        return ((self.L[:, None, :] - self.L[None, :, :]) ** 2).sum(2)


    @property
    def mu_W(self):
        Mu = -self.D + self.b
        Mu = np.tile(Mu[:, :, None], (1, 1, self.B))
        for n in range(self.N):
            Mu[n, n, :] = self._self_gaussian.mu

        return Mu

    @property
    def sigma_W(self):
        sig = self.cov.sigma
        Sig = np.tile(sig[None, None, :, :], (self.N, self.N, 1, 1))

        for n in range(self.N):
            Sig[n, n, :, :] = self._self_gaussian.sigma

        return Sig

    def initialize_from_prior(self):
        self.L = np.sqrt(self.eta) * np.random.randn(self.N, self.dim)
        self.cov.resample()

    def initialize_hypers(self, W):
        # Optimize the initial locations
        self._optimize_L(np.ones((self.N, self.N)), W)

    def _hmc_log_probability(self, L, b, A, W):
        """
        Compute the log probability as a function of L.
        This allows us to take the gradients wrt L using autograd.
        :param L:
        :param A:
        :return:
        """
        assert self.B == 1
        import autograd.numpy as atnp

        # Compute pairwise distance
        L1 = atnp.reshape(L, (self.N, 1, self.dim))
        L2 = atnp.reshape(L, (1, self.N, self.dim))
        # Mu = a * anp.sqrt(anp.sum((L1-L2)**2, axis=2)) + b
        Mu = -atnp.sum((L1 - L2) ** 2, axis=2) + b

        Aoff = A * (1 - atnp.eye(self.N))
        X = (W - Mu[:, :, None]) * Aoff[:, :, None]

        # Get the covariance and precision
        Sig = self.cov.sigma[0, 0]
        Lmb = 1. / Sig

        lp = atnp.sum(-0.5 * X ** 2 * Lmb)

        # Log prior of L under spherical Gaussian prior
        lp += -0.5 * atnp.sum(L * L / self.eta)

        # Log prior of mu0 under standardGaussian prior
        lp += -0.5 * b ** 2

        return lp

    def resample(self, data=[]):
        super(_LatentDistanceModelGaussianMixin, self).resample(data)
        A, W = data
        N, B = self.N, self.B
        self._resample_L(A, W)
        self._resample_b(A, W)
        self._resample_cov(A, W)
        self._resample_self_gaussian(A, W)
        self._resample_eta()
        # print "eta: ", self.eta, "\tb: ", self.b

    def _resample_L(self, A, W):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc

        lp = lambda L: self._hmc_log_probability(L, self.b, A, W)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        # lp0 = lp(self.L)
        self.L = hmc(lp, dlp, stepsz, nsteps, self.L.copy(), negative_log_prob=False)
        # lpf = lp(self.L)
        # print "diff lp: ", (lpf - lp0)

    def _optimize_L(self, A, W):
        """
        Resample the locations given A
        :return:
        """
        import autograd.numpy as atnp
        from autograd import grad
        from scipy.optimize import minimize

        lp = lambda Lflat: -self._hmc_log_probability(atnp.reshape(Lflat, (self.N, 2)), self.b, A, W)
        dlp = grad(lp)

        res = minimize(lp, np.ravel(self.L), jac=dlp, method="bfgs")

        self.L = np.reshape(res.x, (self.N, 2))

    def _resample_b_hmc(self, A, W):
        """
        Resample the distance dependence offset
        :return:
        """
        # TODO: We could sample from the exact Gaussian conditional
        from autograd import grad
        from hips.inference.hmc import hmc

        lp = lambda b: self._hmc_log_probability(self.L, b, A, W)
        dlp = grad(lp)

        stepsz = 0.0001
        nsteps = 10
        b = hmc(lp, dlp, stepsz, nsteps, np.array(self.b), negative_log_prob=False)
        self.b = float(b)
        print("b: ", self.b)


    def _resample_b(self, A, W):
        """
        Resample the distance dependence offset
        W ~ N(mu, sigma)
          = N(-D + b, sigma)
    
        implies
        W + D ~ N(b, sigma).
    
        If b ~ N(0, 1), we can compute the Gaussian conditional
        in closed form.
        """
        D = self.D
        sigma = self.cov.sigma[0, 0]
        Aoff = (A * (1 - np.eye(self.N))).astype(np.bool)
        X = (W + D[:, :, None])[Aoff]

        # Now X ~ N(b, sigma)
        mu0, sigma0 = 0.0, 1.0
        N = X.size
        sigma_post = 1. / (1. / sigma0 + N / sigma)
        mu_post = sigma_post * (mu0 / sigma0 + X.sum() / sigma)

        self.b = mu_post + np.sqrt(sigma_post) * np.random.randn()
        # print "b: ", self.b


    def _resample_cov(self, A, W):
        # Resample covariance matrix
        Mu = self.Mu
        mask = (True - np.eye(self.N, dtype=np.bool)) & A.astype(np.bool)
        self.cov.resample(W[mask] - Mu[mask])


    def _resample_self_gaussian(self, A, W):
        # Resample self connection
        mask = np.eye(self.N, dtype=np.bool) & A.astype(np.bool)
        self._self_gaussian.resample(W[mask])


    def _resample_eta(self):
        """
        Resample sigma under an inverse gamma prior, sigma ~ IG(1,1)
        :return:
        """
        L = self.L

        a_prior = 1.0
        b_prior = 1.0

        a_post = a_prior + L.size / 2.0
        b_post = b_prior + (L ** 2).sum() / 2.0

        from scipy.stats import invgamma
        self.eta = invgamma.rvs(a=a_post, scale=b_post)
        # print "eta: ", self.eta


### Adjacency models
class _FixedAdjacencyMixin(_NetworkModel):
    def __init__(self, T, N, B, rho=0.5, rho_self=None, **kwargs):
        super(_FixedAdjacencyMixin, self).__init__(T, N, B)
        self._rho = expand_scalar(rho, (N, N))
        if rho_self is not None:
            self._rho[np.diag_indices(N)] = rho_self

    @property
    def rho(self):
        return self._rho

    def resample(self,data=[]):
        super(_FixedAdjacencyMixin, self).resample(data)


class _DenseAdjacencyMixin(_NetworkModel):
    def __init__(self, T, N, B, **kwargs):
        super(_DenseAdjacencyMixin, self).__init__(T, N, B)
        self._rho = np.ones((N,N))

    @property
    def rho(self):
        return self._rho

    def resample(self,data=[]):
        super(_DenseAdjacencyMixin, self).resample(data)


class _IndependentBernoulliMixin(_NetworkModel):

    def __init__(self, N, B,
                 a_0=1.0, b_0=1.0,
                 is_diagonal_conn_special=True,
                 **kwargs):
        super(_IndependentBernoulliMixin, self).__init__(N, B)
        raise NotImplementedError("TODO: Implement the BetaBernoulli class")

        assert np.isscalar(a_0)
        assert np.isscalar(b_0)
        self._betabernoulli = BetaBernoulli(a_0, b_0)

        self.is_diagonal_conn_special = is_diagonal_conn_special
        if is_diagonal_conn_special:
            self._self_betabernoulli = BetaBernoulli(a_0, b_0)

    @property
    def rho(self):
        N, B = self.N, self.B
        rho = np.zeros((N, N))
        if self.is_diagonal_conn_special:
            # Set off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            rho[mask] = self._betabernoulli.rho

            # set diagonal weights
            mask = np.eye(N).astype(bool)
            rho[mask] = self._self_betabernoulli.rho

        else:
            rho = self._betabernoulli.rho * np.ones((N, N))
        return rho

    def resample(self, data=[]):
        super(_IndependentBernoulliMixin, self).resample(data)
        A, W = data
        N, B = self.N, self.B
        if self.is_diagonal_conn_special:
            # Resample prior for off-diagonal conns
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            self._betabernoulli.resample(A[mask])

            # Resample prior for off-diagonal conns
            mask = np.eye(N).astype(bool)
            self._self_betabernoulli.resample(A[mask])

        else:
            # Resample prior for all conns
            mask = np.ones((N, N), dtype=bool)
            self._betabernoulli.resample(A[mask])

# TODO: Define the distance and block models

### Define different combinations of network models
class FixedMeanDenseNetwork(_DenseAdjacencyMixin,
                            _FixedWeightsMixin):
    pass

class FixedMeanSparseNetwork(_FixedAdjacencyMixin,
                             _FixedWeightsMixin):
    pass
