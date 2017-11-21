#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:55:59 2017

@author: roger
"""

"""
The "generalized linear models" of computational neuroscience
are ultimately nonlinear vector autoregressive models in
statistics. As the name suggests, the key component in these
models is a regression from inputs, x, to outputs, y.

When the outputs are discrete random variables, like spike
counts, we typically take the regression to be a generalized
linear model:

   y ~ p(mu(x), theta)
   mu(x) = f(w \dot x)

where 'p' is a discrete distribution, like the Poisson,
and 'f' is a "link" function that maps a linear function of
x to the parameters of 'p'. Hence the name "GLM" in
computational neuroscience.

Our contribution is a host of hierarchical models for the
weights of the GLM, along with an efficient Bayesian inference
algorithm for inferring the weights, 'w', under count observations.
Specifically, we build hierarchical sparse priors for the weights
and then leverage Polya-gamma augmentation to perform efficient
inference.

This module implements these sparse regressions.
"""
import abc
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import numpy.random as npr

from scipy.linalg import block_diag
from scipy.linalg.lapack import dpotrs
from scipy import signal
from sksparse.cholmod import cholesky
from pybasicbayes.abstractions import GibbsSampling
from pybasicbayes.util.stats import sample_gaussian, sample_discrete_from_log, sample_invgamma

from pyglm.utils.utils import logistic, expand_scalar, expand_cov


class _SparseScalarRegressionBase(GibbsSampling):
    """
    Base class for the sparse regression.

    We assume the output dimension D = 1

    N: number of input groups
    B: input dimension for each group
    N0: N*T
    inputs: X \in R^{N \times B}
    outputs: y \in R^D

    model:

    y_d = \sum_{n=1}^N a_{d,n} * (w_{d,n} \dot x_n) + b_d + noise

    where:

    a_n \in {0,1}      is a binary indicator
    w_{d,n} \in R^B    is a weight matrix for group n
    x_n \in R^B        is the input for group n
    b \in R^D          is a bias vector

    hyperparameters:

    rho in [0,1]^N     probability of a_n for each group n
    mu_w in R^{DxTxNxB}  mean of weight matrices
    S_w in R^{DxTxNxBxB} covariance for each row of the the weight matrices
    mu_b in R^D        mean of the bias vector
    S_b in R^{DxD}     covariance of the bias vector

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, T, N, B,
                 rho=0.5,
                 mu_w=0.0, S_w=1.0,
                 mu_b=0.0, S_b=1.0):
        self.T, self.N, self.B = T, N, B

        # Initialize the hyperparameters
        self.rho = rho
        self.mu_w = mu_w
        self.mu_b = mu_b
        self.S_w = S_w
        self.S_b = S_b
        self.A  = 1

        # Initialize the model parameters with a draw from the prior
        self.a = npr.rand(self.N) < self.rho
        self.W = np.zeros((T, N, B))

        for n in range(self.N):
            self.W[0, n,] = npr.multivariate_normal(self.mu_w[0,n,], self.S_w[0, 0,])
        for t in range(self.T - 1):
            for n in range(self.N):
                self.W[t + 1, n,] = self.a[n] * npr.multivariate_normal(self.A * self.W[t, n,], self.S_w[t, n,])

        for n in range(self.N):
            self.W[:, n, 0] = scipy.signal.savgol_filter(self.W[:, n, 0], 101, 1)

        # sine wave
        # Fs = 3000
        # f = 5
        # sample = T
        # x = np.arange(sample)
        # for n in range(self.N):
        #     self.W[:,n,0] = 0.75*np.cos(2 * np.pi * f * x / Fs + np.pi)+ 0.75

        # square wave
        #for n in range(self.N):
        #    self.W[:,n,0] = - 0.55*signal.square(2 * np.pi * 5 * np.arange(T)/ 5000) + 0.55

        # Triangle wave
        # t = np.linspace(0, 1, T)
        # for n in range(self.N):
            #self.W[:,n,0] = 0.75 * signal.sawtooth(2 * np.pi * 3 * t) + 0.75

        self.b = npr.multivariate_normal(self.mu_b, self.S_b)

    # Properties
    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = expand_scalar(value, (self.N,))

    @property
    def mu_w(self):
        return self._mu_w

    @mu_w.setter
    def mu_w(self, value):
        T, N, B = self.T, self.N, self.B
        #self._mu_w = expand_scalar(value, (T, N, B))
        mu0 = 0 * np.random.rand(N, B)
        mu = np.repeat(mu0, T, axis=1)
        self._mu_w = mu.transpose().reshape((T,N,B))

    @property
    def mu_b(self):
        return self._mu_b

    @mu_b.setter
    def mu_b(self, value):
        self._mu_b = expand_scalar(value, (1,))

    @property
    def S_w(self):
        return self._S_w

    @S_w.setter
    def S_w(self, value):
        T, N, B = self.T, self.N, self.B
        self._S_w = expand_cov(value, (T, N, B, B))

    @property
    def S_b(self):
        return self._S_b

    @S_b.setter
    def S_b(self, value):
        assert np.isscalar(value)
        self._S_b = expand_cov(value, (1, 1))

    @property
    def natural_params(self):
        # Compute information form parameters
        T, N, B = self.T, self.N, self.B
        J_w = np.zeros((T * N, B, B))
        h_w = np.zeros((T * N, B))
        mu_w1 = np.zeros((T, N, B))
        for t in range(T):
            mu_w1[t, :, :] = pow(self.A, t) * self.mu_w[t, :, :]
        for n in range(T * N):
            J_w[n] = np.linalg.inv(self.S_w.reshape(T * N, B, B)[n])
            # to do implementation because current intitial mu = 0
            h_w[n] = J_w[n].dot(np.transpose(mu_w1.reshape(T * N, B)[n]))
            #h_w[n] = J_w[n].dot(np.transpose(self.mu_w.reshape(T * N, B)[n]))

        J_b = np.linalg.inv(self.S_b)
        h_b = J_b.dot(self.mu_b)

        return J_w, h_w, J_b, h_b

    @property
    def deterministic_sparsity(self):
        return np.all((self.rho < 1e-6) | (self.rho > 1 - 1e-6))

    @abc.abstractmethod
    def omega(self, X, y):
        """
        The "precision" of the observations y. For the standard
        homoskedastic Gaussian model, this is a function of model parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def kappa(self, X, y):
        """
        The "normalized" observations, y. For the standard
        homoskedastic Gaussian model, this is the data times the precision.
        """
        raise NotImplementedError

    def _flatten_X(self, X):
        if X.ndim == 2:
            assert X.shape[1] == self.N * self.B
        elif X.ndim == 3:
            X = np.reshape(X, (-1, self.N * self.B))
        else:
            raise Exception
        return X

    def extract_data(self, data):
        N, B = self.N, self.B

        assert isinstance(data, tuple) and len(data) == 2
        X, y = data
        T = X.shape[0]
        assert y.shape == (T, 1) or y.shape == (T,)

        # Reshape X such that it is T x NB
        X = self._flatten_X(X)
        return X, y

    def activation(self, X):
        N, B, T = self.N, self.B, self.T
        X = self._flatten_X(X)
        # need implemenation more. This is not correct

        W = np.reshape((self.a[:, None] * self.W), (T * N * B,))
        b = self.b[0]
        # b = self.b
        # new form of X; old form: return X.dot(W) + b
        return block_diag(*X).dot(W) + b

    @abc.abstractmethod
    def mean(self, X):
        """
        Return the expected value of y given X.
        """
        raise NotImplementedError

    @property
    def _prior_sufficient_statistics(self):
        """
        Compute the prior statistics (information form Gaussian
        potentials) for the complete set of weights and biases.
        
        Modified version for Gaussian
        """
        T, N, B = self.T, self.N, self.B
        J_w, h_w, J_b, h_b = self.natural_params

        J_w1 = J_w.reshape(T*N*B,)
        # J_w = J_w.reshape(N * T * B,)
        J_b1 = J_b.reshape(B,)
        #
        J_prior_ofdiag = -self.A * J_w1[:-N * B]
        #J_prior_ofdiag = -0.9*J_w1[:-N * B]
        J_prior_diag = self.A * self.A * J_w1 + J_w1
        #J_prior_diag = 0.9*0.9*J_w1 + J_w1
        J_prior_diag[-N * B:] = J_w1[0:N * B]
        J_prior_diag = np.append(J_prior_diag, J_b1)
        #
        data = np.zeros((3, N*T*B+1))
        data[0, :-N * B - 1] = J_prior_ofdiag
        data[2, N*B : -1] = J_prior_ofdiag
        data[1, ] = J_prior_diag
        #

        J_prior = scipy.sparse.spdiags(data, np.array([-N * B, 0, N * B]), N * B * T + 1, N * B * T + 1, format = 'csc')

        # Z = np.zeros((n, n))  # zeros
        # To debug here because J_w is not correct


        # B0 = np.diag([1]*T) + np.diag([2]*np.ones(T-1), 1) + np.diag(2*np.ones(T-1), -1)
        # # B = array([[ 1.,  2.,  0.],
        # #            [ 2.,  1.,  2.],
        # #            [ 0.,  2.,  1.]])
        # # build 2d list `Bm` replacing 0->Z, 1->T, 2->I:
        # bdict = {0.:Z, 1.:P, 2.:I}
        # Bm = [[bdict[i] for i in rw] for rw in B0]
        # # Use the power of bmat to construct matrix:
        # J_prior0 = np.asarray(np.bmat(Bm))
        # J_prior0[-n:,-n:] = block_diag(*(J_w[0:n]))
        #
        # J_prior = block_diag(J_prior0, J_b)

        # assert J_prior.shape == (T * N * B + 1, T * N * B + 1)
        # assert J_prior.shape == (T * N * B, T * N * B)

        # h_prior = np.concatenate((h_w.ravel(), h_b.ravel()))
        # mu_w1 = np.zeros((T, N, B))

        # to debug bucause current initial value of mu = 0
        # for t in range(T):
        #     mu_w1[t, :, :] = 0.99 ** t * self.mu_w[t, :, :]
        # h_prior0 = J_prior0.dot(mu_w1.reshape(T * N * B))
        # h_prior = np.concatenate((h_prior0.ravel(), h_b.ravel()))
        #

        h_prior0 = scipy.sparse.csc_matrix(h_w)
        #h_prior0 = scipy.sparse.csc_matrix((T*N*B,1))
        h_prior  = scipy.sparse.vstack([h_prior0,h_b], format='csc')
        # assert h_prior.shape == (T * N * B + 1,)
        # assert h_prior.shape == (T * N * B,)
        return J_prior, h_prior

    def _lkhd_sufficient_statistics(self, datas):
        """
        Compute the likelihood statistics (information form Gaussian
        potentials) for each dataset.  Polya-gamma regressions will
        have to override this class.
        """
        T, N, B = self.T, self.N, self.B

        #J_lkhd = np.zeros((T * N * B + 1, T * N * B + 1))
        h_lkhd0 = np.zeros((T * N * B + 1, 1))

        # Compute the posterior sufficient statistics
        for data in datas:
            assert isinstance(data, tuple)
            # Dimension of X: T*NB
            X, y = self.extract_data(data)
            T = X.shape[0]

            # Get the precision and the normalized observations
            omega = self.omega(X, y)
            assert omega.shape == (T,)
            kappa = self.kappa(X, y)
            assert kappa.shape == (T,)

            # Add the sufficient statistics to J_lkhd
            # The last row and column correspond to the
            # affine term

            # Sparse Version 2017-05-04
            # X0 = scipy.sparse.block_diag(X).multiply(omega[:, None])
            # J_lkhd0 = X0.transpose().dot(scipy.sparse.block_diag(X))
            # J_lkhd  = scipy.sparse.bmat([[J_lkhd0, X0.sum(0).T], [X0.sum(0), omega.sum()]])

            # Sparse Version 2017-05-05
            X_all = np.zeros((T,N*B,N*B))

            for i in range(T):
                X1 = X[i, :] * omega[i]
                X_all[i, :, :] = X[i, :][:, None].dot(X1[None, :])
            # Sparse version 2017-05-31
            # J_lkhd0 = np.zeros((N*B*T+1, N*B*T+1))

            # X_m = np.reshape(X * omega[:,None],(N*T*B,))
            # J_lkhd0[:T * N * B,:T * N * B] = block_diag(*X_all)
            # J_lkhd0[:T * N * B, -1] += X_m
            # J_lkhd0[-1, :T * N * B] += X_m
            # J_lkhd0[-1, -1] += omega.sum()
            # J_lkhd = scipy.sparse.csc_matrix(J_lkhd0)

            # New_version 2017-08-03
            X_m = np.reshape(X * omega[:, None], (N * T * B,))
            S = scipy.sparse.block_diag(X_all)
            S1 = scipy.sparse.hstack((S, X_m[:, None]))
            X_m1 = np.append(X_m[:, None], omega.sum())
            J_lkhd = scipy.sparse.vstack((S1, X_m1[None, :]), 'csc')

            #J_lkhd  = scipy.sparse.bmat([[J_lkhd0, X_m], [X_m.T, omega.sum()]])
            # XO = block_diag(*X) * omega[:, None]
            # J_lkhd[:T * N * B, :T * N * B] += XO.T.dot(block_diag(*X))
            # Xsum = XO.sum(0)
            # J_lkhd[:T * N * B, -1] += Xsum
            # J_lkhd[-1, :T * N * B] += Xsum
            # J_lkhd[-1, -1] += omega.sum()

            # Add the sufficient statistics to h_lkhd
            # h_lkhd0 = scipy.sparse.csc_matrix(kappa).dot(scipy.sparse.block_diag(X))
            # h_lkhd1 = scipy.sparse.hstack([h_lkhd0, kappa.sum()], format = 'csc')
            # h_lkhd  = h_lkhd1.transpose()

            # Add the sufficient statistics to h_lkhd  2017-05-05
            X_m1 = np.reshape(X * kappa[:, None], (N * T * B, 1))
            h_lkhd0 = np.append(X_m1, kappa[:,None].sum())
            h_lkhd = scipy.sparse.csc_matrix(h_lkhd0)
            h_lkhd = h_lkhd.transpose()

        return J_lkhd, h_lkhd

    ### Gibbs sampling
    def resample(self, datas):
        # Compute the prior and posterior sufficient statistics of W
        J_prior, h_prior = self._prior_sufficient_statistics
        J_lkhd, h_lkhd = self._lkhd_sufficient_statistics(datas)

        J_post = J_prior + J_lkhd
        h_post = h_prior + h_lkhd

        # Resample a
        # if self.deterministic_sparsity:
        self.a = np.round(self.rho).astype(bool)
        # else:
        # self._collapsed_resample_a(J_prior, h_prior, J_post, h_post)

        # Resample weights
        self._resample_W(J_post, h_post)

    def _collapsed_resample_a(self, J_prior, h_prior, J_post, h_post):
        """
        """
        N, B, rho = self.N, self.B, self.rho
        perm = npr.permutation(self.N)

        ml_prev = self._marginal_likelihood(J_prior, h_prior, J_post, h_post)
        for n in perm:
            # TODO: Check if rho is deterministic

            # Compute the marginal prob with and without A[m,n]
            lps = np.zeros(2)
            # We already have the marginal likelihood for the current value of a[m]
            # We just need to add the prior
            v_prev = int(self.a[n])
            lps[v_prev] += ml_prev
            lps[v_prev] += v_prev * np.log(rho[n]) + (1 - v_prev) * np.log(1 - rho[n])

            # Now compute the posterior stats for 1-v
            v_new = 1 - v_prev
            self.a[n] = v_new

            ml_new = self._marginal_likelihood(J_prior, h_prior, J_post, h_post)

            lps[v_new] += ml_new
            lps[v_new] += v_new * np.log(rho[n]) + (1 - v_new) * np.log(1 - rho[n])

            # Sample from the marginal probability
            # max_lps = max(lps[0], lps[1])
            # se_lps = np.sum(np.exp(lps-max_lps))
            # lse_lps = np.log(se_lps) + max_lps
            # ps = np.exp(lps - lse_lps)
            # v_smpl = npr.rand() < ps[1]å
            v_smpl = sample_discrete_from_log(lps)
            self.a[n] = v_smpl

            # Cache the posterior stats and update the matrix objects
            if v_smpl != v_prev:
                ml_prev = ml_new

    def _resample_W(self, J_post, h_post):
        """
        Resample the weight of a connection (synapse)
        """
        T, N, B = self.T, self.N, self.B
        # repeat: element is copied each time
        # a0 = np.concatenate((np.tile(np.repeat(self.a, self.B), self.T), [1])).astype(np.bool)
        # a = np.tile(np.repeat(self.a, self.B), self.T)
        # a = np.tile(np.repeat(self.a, self.B), 2).astype(np.bool)

        # N0 = np.sum(self.a)

        # Jp = J_post[np.ix_(a0, a0)]
        # hp = h_post[a0]

        # get sparse tridiagonal matrix
        # a1 = np.zeros((N0 * B * T + 1, 2 * N0 * B + 1))

        # a1 = np.zeros((N0*B*T, 2*N0*B+1))
        # b = list(range(-N0*B, N0*B+1))
        # for i in range(2*N0*B+1):
        #    a1[:, i] = np.append(np.diag(Jp, i - N0*B), [0] * abs(b[i]))

        # J2 = scipy.sparse.spdiags(a1.T, list(range(-N0 * B, N0 * B + 1)), N0 * B * T, N0 * B * T, format='coo')
        # J3 = scipy.sparse.hstack((J2,np.transpose(scipy.sparse.coo_matrix(J_post[:-1,-1]))),format='coo')
        # J4 = scipy.sparse.vstack((J3,scipy.sparse.coo_matrix(J_post[-1,:])),format='csc')

        # Sample in information form
        x = np.random.randn(T*N*B+1)
        # J4 = scipy.sparse.csc_matrix(J_post)
        lp = cholesky(J_post,ordering_method='natural')
        L1 = lp.L()
        # W = scipy.sparse.linalg.spsolve(L1, x).reshape((N*B*T+1,1)) + lp(hp)
        W = scipy.sparse.linalg.spsolve(L1.T, x).reshape(N*B*T+1,1) + lp(h_post)
        W_all = np.asarray(W).reshape(-1)

        # W = sample_gaussian(J=Jp, h=hp)
        # W = sample_sparse_gaussian(J=Jp, h=hp, l1=self.lp)
        # Set bias and weights
        self.W *= 0
        self.W[:, self.a, :] = W_all[:-1].reshape(T, -1, B)
        # self.W = np.reshape(W[:-1], (D,N,B))
        self.b = np.reshape(W_all[-1], (1,))

    def _marginal_likelihood(self, J_prior, h_prior, J_post, h_post):
        """
        Compute the marginal likelihood as the ratio of log normalizers
        """
        # Repeat a with #B basis function, and concatenate
        N, B, T = self.N, self.B, self.T
        a0 = np.concatenate((np.tile(np.repeat(self.a, self.B), self.T), [1])).astype(np.bool)
        a = np.tile(np.repeat(self.a, self.B), self.T)

        # Extract the entries for which A=1
        # "a" is bool type with A=1 having "True"; A=0 having "False"
        # Extract the diagonal entries 
        # J_prior and J_post are dignoal

        J0 = J_prior[np.ix_(a0, a0)]
        h0 = h_prior[a0]
        Jp = J_post[np.ix_(a0, a0)]
        hp = h_post[a0]

        # extract block tridiagonal

        # build sparse block tridiagonal

        # This relates to the mean/covariance parameterization as follows
        # log |C| = log |J^{-1}| = -log |J|
        # and
        # mu^T C^{-1} mu = mu^T h
        #                = mu C^{-1} C h
        #                = h^T C h
        #                = h^T J^{-1} h
        # ml = 0
        # ml -= 0.5*np.linalg.slogdet(Jp)[1]
        # ml += 0.5*np.linalg.slogdet(J0)[1]
        # ml += 0.5*hp.T.dot(np.linalg.solve(Jp, hp))
        # ml -= 0.5*h0.T.dot(np.linalg.solve(J0, h0))

        # for Gaussian case to accelerate computation

        # N0 = np.sum(self.a)

        # a1 = np.zeros((N0 * B * T + 1, 2 * N0 * B + 1))
        # a2 = np.zeros((N0 * B * T, 2 * N0 * B + 1))

        # b = list(range(-N0 * B, N0 * B + 1))
        # for i in range(2 * N0 * B + 1):
        #    a1[:, i] = np.append(np.diag(J0, i - N0 * B), [0] * abs(b[i]))
        #    a2[:, i] = np.append(np.diag(Jp, i - N0 * B), [0] * abs(b[i]))

        # J1 = scipy.sparse.spdiags(a1.T, list(range(-N0 * B, N0 * B + 1)), N0 * B * T +1, N0 * B * T +1, format='csc')

        # J2 = scipy.sparse.spdiags(a2.T, list(range(-N0 * B, N0 * B + 1)), N0 * B * T, N0 * B * T, format='coo')
        # J3 = scipy.sparse.hstack((J2, np.transpose(scipy.sparse.coo_matrix(J_post[:-1, -1]))),format='coo')
        # J4 = scipy.sparse.vstack((J3, scipy.sparse.coo_matrix(J_post[-1, :])),format='csc')


        J1 = scipy.sparse.csc_matrix(J0)
        J4 = scipy.sparse.csc_matrix(Jp)
        l0 = cholesky(J1)
        lp = cholesky(J4)
        ml = 0
        ml -= 0.5 * lp.logdet()
        ml += 0.5 * hp.T.dot(lp(hp))

        ml += 0.5 * l0.logdet()
        ml -= 0.5 * h0.T.dot(l0(h0))

        # Now compute it even faster using the Cholesky!
        # L0 = np.linalg.cholesky(J0)
        # Lp = np.linalg.cholesky(Jp)

        # ml = 0
        # ml -= np.sum(np.log(np.diag(Lp)))
        # ml += 0.5 * hp.T.dot(dpotrs(Lp, hp, lower=True)[0])

        # ml += np.sum(np.log(np.diag(L0)))
        # ml -= 0.5*h0.T.dot(dpotrs(L0, h0, lower=True)[0])

        return ml


class SparseGaussianRegression(_SparseScalarRegressionBase):
    """
    The standard case of a sparse regression with Gaussian observations.
    """

    def __init__(self, N, B,
                 a_0=2.0, b_0=2.0, eta=None,
                 **kwargs):
        super(SparseGaussianRegression, self).__init__(N, B, **kwargs)

        # Initialize the noise model
        assert np.isscalar(a_0) and a_0 > 0
        assert np.isscalar(b_0) and a_0 > 0
        self.a_0, self.b_0 = a_0, b_0
        if eta is not None:
            assert np.isscalar(eta) and eta > 0
            self.eta = eta
        else:
            # Sample eta from its inverse gamma prior
            self.eta = sample_invgamma(self.a_0, self.b_0)

    def log_likelihood(self, x):
        N, B, eta = self.N, self.B, self.eta

        X, y = self.extract_data(x)
        return -0.5 * np.log(2 * np.pi * eta) - 0.5 * (y - self.mean(X)) ** 2 / eta

    def rvs(self, size=[], X=None, psi=None):
        N, B = self.N, self.B

        if psi is None:
            if X is None:
                assert isinstance(size, int)
                X = npr.randn(size, N * B)

            X = self._flatten_X(X)
            psi = self.mean(X)

        return psi + np.sqrt(self.eta) * npr.randn(*psi.shape)

    def omega(self, X, y):
        T = X.shape[0]
        return 1. / self.eta * np.ones(T)

    def kappa(self, X, y):
        return y / self.eta

    def resample(self, datas):
        super(SparseGaussianRegression, self).resample(datas)
        self._resample_eta(datas)

    def mean(self, X):
        return self.activation(X)

    def _resample_eta(self, datas):
        N, B = self.N, self.B

        alpha = self.a_0
        beta = self.b_0
        for data in datas:
            X, y = self.extract_data(data)
            T = X.shape[0]

            alpha += T / 2.0
            beta += np.sum((y - self.mean(X)) ** 2)

        self.eta = sample_invgamma(alpha, beta)


class GaussianRegression(SparseGaussianRegression):
    """
    The standard scalar regression has dense weights.
    """

    def __init__(self, N, B,
                 **kwargs):
        rho = np.ones(N)
        kwargs["rho"] = rho
        super(GaussianRegression, self).__init__(N, B, **kwargs)


class _SparsePGRegressionBase(_SparseScalarRegressionBase):
    """
    Extend the sparse scalar regression to handle count observations
    by leveraging the Polya-gamma augmentation for logistic regression
    models. This supports the subclasses implemented below. Namely:
    - SparseBernoulliRegression
    - SparseBinomialRegression
    - SparseNegativeBinomialRegression
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, T, N, B, **kwargs):
        super(_SparsePGRegressionBase, self).__init__(T, N, B, **kwargs)

        # Initialize Polya-gamma samplers
        import pypolyagamma as ppg
        num_threads = ppg.get_omp_num_threads()
        seeds = npr.randint(2 ** 16, size=num_threads)
        self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

    @abc.abstractmethod
    def a_func(self, y):
        raise NotImplementedError

    @abc.abstractmethod
    def b_func(self, y):
        raise NotImplementedError

    @abc.abstractmethod
    def c_func(self, y):
        raise NotImplementedError

    def log_likelihood(self, x):
        X, y = self.extract_data(x)
        psi = self.activation(X)
        return np.log(self.c_func(y)) + self.a_func(y) * psi - self.b_func(y) * np.log1p(np.exp(psi))

    def omega(self, X, y):
        """
        In the Polya-gamma augmentation, the precision is
        given by an auxiliary variable that we must sample
        """
        import pypolyagamma as ppg
        psi = self.activation(X)
        omega = np.zeros(y.size)
        ppg.pgdrawvpar(self.ppgs,
                       self.b_func(y).ravel(),
                       psi.ravel(),
                       omega)
        return omega.reshape(y.shape)

    def kappa(self, X, y):
        return self.a_func(y) - self.b_func(y) / 2.0


class SparseBernoulliRegression(_SparsePGRegressionBase):
    def a_func(self, data):
        return data

    def b_func(self, data):
        return np.ones_like(data, dtype=np.float)

    def c_func(self, data):
        return 1.0

    def mean(self, X):
        psi = self.activation(X)
        return logistic(psi)

    def rvs(self, X=None, size=[], psi=None):
        if psi is None:
            if X is None:
                assert isinstance(size, int)
                X = npr.randn(size, self.N * self.B)

            X = self._flatten_X(X)
            p = self.mean(X)
        else:
            p = logistic(psi)

        y = npr.rand(*p.shape) < p

        return y


class BernoulliRegression(SparseBernoulliRegression):
    """
    The standard Bernoulli regression has dense weights.
    """

    def __init__(self, N, B, **kwargs):
        rho = np.ones(N)
        kwargs["rho"] = rho
        super(BernoulliRegression, self).__init__(N, B, **kwargs)
