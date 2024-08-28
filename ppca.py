import numpy as np


class PPCA:
    """Probabilistic Principle Component Analysis (PPCA).
    PPCA is a probabilistic latent variable model, whose maximum likelihood
    solution corresponds to PCA. For an introduction to PPCA, see [1].

    This implementation uses the expectation-maximization (EM) algorithm to
    find maximum-likelihood estimates of the PPCA model parameters. This
    enables a principled handling of missing values in the dataset, assuming
    that the values are missing at random.

      p(z) = N(0, I)
    p(x|z) = N(W_ z + mean_, noise_variance_ I)


    Attributes:

    W_ : (ndarray of shape (n_components, n_features))
        Principal axes in feature space, representing the
        directions of maximum variance in the data, scaled
        by the square root of a noise-adjusted variance
        parameter. The components are sorted by decreasing
        explained_variance_.

    mean_ : (ndarray of shape (n_features,))
        Per-feature empirical mean, estimated from the training set.

    noise_variance_ : (float)
        The estimated noise variance of the conditional distribution.

    explained_variance_ : (ndarray of shape (n_components,))
        The amount of variance explained by each of the
        selected components. The variance estimation
        uses n_samples degrees of freedom.

    explained_variance_ratio_ : (ndarray of shape (n_components,))
        Percentage of variance explained by each
        of the selected components.

    n_components_ : (int)
        The number of components.

    n_samples_ : (int)
        Number of samples in the training data.

    n_features_in_ : (int)
        Number of features seen during fit.

    References:
    [1] Bishop, C. M., Pattern Recognition and Machine Learning. New York:
        Springer, 2006.
    [2] Tipping, M. E. and Bishop, C. M., Probabilistic Principal Component
        Analysis. Journal of the Royal Statistical Society: Series B
        (Statistical Methodology), 1999.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 10000,
        min_iter: int = 20,
        rtol: float = 1e-6,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        """Initialize a PPCA model.

        Parameters:

        n_components : (int)
            Number of components to estimate. Has to be smaller than
            min(n_features, n_samples).
        max_iter : (int)
            Maximum number of iterations to run EM algorithm.
        min_iter : (int)
            Minimum number of iterations to run EM algorithm.
        rtol : (float)
            Convergence criterion for EM: relative change in parameters
            and complete-data negative log likelihood.
        verbose : (bool)
            Whether to print convergence info.
        random_state : (int)
            Fix a random seed to replicate EM initial conditions.
        """
        if rtol <= 0:
            raise ValueError("rtol must be greater than zero")

        self.n_components_ = n_components
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.rtol = rtol
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray):
        """Maximum likelihood estimation of model parameters using
        the EM algorithm.

        Parameters:

        X : (ndarray of shape (n_samples, n_features))
            observations

        Returns:

        ppca : (PPCA)
            fitted PPCA model
        """
        # code expects X as (n_features, n_samples)
        X = X.T

        self.n_features_in_, self.n_samples_ = X.shape
        max_rank = min(self.n_features_in_, self.n_samples_)
        if self.n_components_ > max_rank - 1:
            raise ValueError(
                "For PPCA, n_components (= {0}) has to be smaller than min(n_samples, n_features) (= {1}).".format(
                    self.n_components_, max_rank
                )
            )

        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError(
                "X contains observations which are entirely NaN. Please remove those before running PPCA."
            )

        W, mu, sig2 = self._em(X)

        # rotate W to the standard PCA basis
        PC, scale, _ = np.linalg.svd(W, full_matrices=False)
        # the largest element in each principle component will have a positive sign
        max_abs_u_cols = np.argmax(np.abs(PC), axis=0)
        shift = np.arange(PC.shape[1])
        indices = max_abs_u_cols + shift * PC.shape[0]
        signs = np.sign(np.take(np.reshape(PC.T, (-1,)), indices, axis=0))
        PC *= signs[np.newaxis, :]
        W = PC @ np.diag(scale)

        # in scikit-learn the variance estimation uses n_samples - 1 degrees of freedom
        # but here we use n_samples degrees of freedom
        self.explained_variance_ = scale**2 + sig2
        self.explained_variance_ratio_ = self.explained_variance_ / (
            sum(self.explained_variance_)
            + sig2 * (self.n_features_in_ - self.n_components_)
        )

        self.W_ = W.T
        self.mean_ = mu
        self.noise_variance_ = sig2

        return self

    def transform(self, X: np.ndarray, return_cov=False, noise_free=False):
        """Transform observations by computing their latent variable
        posterior.

        Parameters:

        X : (ndarray of shape (n_samples, n_features))
            observations
        return_cov : (bool)
            whether to return the covariance matrix
        noise_free : (bool)
            whether to assume sig2 = 0

        Returns:

        Z : (ndarray of shape (n_samples, n_components))
            mean of latent variables
        covZ : (ndarray of shape (n_samples, n_components, n_components)), optional
            covariance matrix of latent variables
        """
        # code expects X as (n_features, n_samples)
        X = X.T
        W = self.W_.T
        sig2 = self.noise_variance_ if not noise_free else 0
        Xc = X - self.mean_[:, None]

        obs = ~np.isnan(Xc)
        if np.all(obs):
            # columns of W are orthogonal
            M_inv = np.diag(1 / (np.diag(W.T @ W) + sig2))
            Z = (M_inv @ W.T @ Xc).T
            covZ = np.tile(sig2 * M_inv[None, :, :], (X.shape[1], 1, 1))
        else:
            Z = np.zeros((self.n_samples_, self.n_components_))
            covZ = np.zeros((self.n_samples_, self.n_components_, self.n_components_))
            for n in range(self.n_samples_):
                w = W[obs[:, n], :]
                m_inv = np.linalg.inv(sig2 * np.eye(self.n_components_) + w.T @ w)
                Z[n, :] = m_inv @ w.T @ Xc[obs[:, n], n]
                covZ[n, :, :] = sig2 * m_inv
        if return_cov:
            return Z, covZ
        return Z

    def fit_transform(self, X: np.ndarray, return_cov=False, noise_free=False):
        """Combine `fit` and `transform`.

        Parameters:

        X : (ndarray of shape (n_samples, n_features))
            observations
        return_cov : (bool)
            whether to return the covariance matrix
        noise_free : (bool)
            whether to assume sig2 = 0

        Returns:

        Z : (ndarray of shape (n_samples, n_components))
            mean of latent variables
        covZ : (ndarray of shape (n_samples, n_components, n_components)), optional
            covariance matrix of latent variables
        """
        return self.fit(X).transform(X, return_cov, noise_free)

    def log_likelihood(self, X, reduction="sum"):
        """Compute the data log likelihood.

        Parameters:

        X : (ndarray of shape (n_samples, n_features))
            observations
        reduction : (str)
            reduce option for log likelihood

        Returns:

        ll : (ndarray of shape (n_samples,) or float)
            data log likelihood
        """
        # code expects X as (n_features, n_samples)
        X = X.T
        W = self.W_.T
        sig2 = self.noise_variance_
        Xc = X - self.mean_[:, None]
        ll = -self._compute_nll(Xc, W, sig2)
        if reduction == "sum":
            return sum(ll)
        return ll

    def reconstruct(self, Z, orthogonal=False):
        """Reconstruct observations from latent variables.

        Parameters:

        Z : (ndarray of shape (n_samples, n_components))
            latent variables
        orthogonal : (bool)
            whether to reconstruct using orthogonal projection

        Returns:

        X_hat : (ndarray of shape (n_samples, n_features))
            reconstructed observations
        """
        # code expects Z as (n_components, n_samples)
        Z = Z.T
        W = self.W_.T
        sig2 = self.noise_variance_
        if orthogonal:
            # optimal reconstruction in the squared reconstruction error sense
            # for the case sig2 > 0
            X_hat = (
                W @ np.diag(1 / np.diag(W.T @ W)) @ np.diag(np.diag(W.T @ W) + sig2) @ Z
                + self.mean_[:, None]
            ).T
        else:
            X_hat = (W @ Z + self.mean_[:, None]).T
        return X_hat

    def _em(self, X: np.ndarray):
        """Expectation-maximization (EM) algorithm for PPCA.

        Parameters:

        X : (ndarray of shape (n_samples, n_features))
            observations

        Returns:

        W : (ndarray of shape (n_features, n_components))
            transformation matrix
        mu : (ndarray of shape (n_features,))
            mean vector
        sig2 : (float)
            variance parameter
        """
        W = np.random.randn(self.n_features_in_, self.n_components_)
        mu = np.nanmean(X, axis=1)
        sig2 = np.random.randn()
        nll = np.inf

        Xc = X - mu[:, None]

        if self.verbose:
            print("{0: >11} {1: >27}".format("Iteration", "Negative log-likelihood"))
            strfmt = "{0: >11d} {1: >27f}"

        itercount = 0
        while itercount < self.max_iter:
            itercount += 1

            Ez, Ezz = self._e_step(Xc, W, sig2)
            W_new, sig2_new = self._m_step(Xc, Ez, Ezz)

            nll_new = sum(self._compute_nll(Xc, W_new, sig2_new))

            if self.verbose and (itercount % 20 == 0):
                print(strfmt.format(itercount, nll_new))

            if itercount > self.min_iter:
                if np.allclose(W_new, W, rtol=self.rtol, atol=0) and np.allclose(
                    sig2_new, sig2, rtol=self.rtol, atol=0
                ):
                    break
                elif np.allclose(nll_new, nll, rtol=self.rtol, atol=0):
                    break

            W = W_new
            sig2 = sig2_new
            nll = nll_new

            if itercount == self.max_iter:
                raise RuntimeError(
                    "Maximum number of iterations reached without convergence: {}".format(
                        self.max_iter
                    )
                )

        return W_new, mu, sig2_new

    def _e_step(
        self,
        Xc: np.ndarray,
        W: np.ndarray,
        sig2: float,
    ):
        """Expectation step of EM algorithm for PPCA.

        Parameters:

        Xc : (ndarray of shape (n_samples, n_features))
            centered observations
        W : (ndarray of shape (n_features, n_components))
            transformation matrix
        sig2 : (float)
            variance parameter

        Returns:

        Ez : (ndarray of shape (n_components, n_samples))
            posterior mean of latent variables
        Ezz : (ndarray of shape (n_components, n_components, n_samples))
            posterior covariance matrix of latent variables
        """
        obs = ~np.isnan(Xc)
        if np.all(obs):
            M_inv = np.linalg.inv(W.T @ W + sig2 * np.eye(self.n_components_))
            Ez = M_inv @ W.T @ Xc
            Ezz = sig2 * M_inv[:, :, None] + Ez[:, None, :] * Ez[None, :, :]
        else:
            Ez = np.zeros((self.n_components_, self.n_samples_))
            Ezz = np.zeros((self.n_components_, self.n_components_, self.n_samples_))
            for n in range(self.n_samples_):
                w = W[obs[:, n], :]
                m_inv = np.linalg.inv(sig2 * np.eye(self.n_components_) + w.T @ w)
                Ez[:, n] = m_inv @ w.T @ Xc[obs[:, n], n]
                Ezz[:, :, n] = sig2 * m_inv + Ez[:, None, n] * Ez[None, :, n]
        return Ez, Ezz

    def _m_step(
        self,
        Xc: np.ndarray,
        Ez: np.ndarray,
        Ezz: np.ndarray,
    ):
        """Maximization step of EM algorithm for PPCA.

        Parameters:

        Xc : (ndarray of shape (n_samples, n_features))
            centered observations
        Ez : (ndarray of shape (n_components, n_samples))
            posterior mean of latent variables
        Ezz : (ndarray of shape (n_components, n_components, n_samples))
            posterior covariance matrix of latent variables

        Returns:

        W : (ndarray of shape (n_features, n_components))
            transformation matrix
        sig2 : (float)
            variance parameter
        """
        obs = ~np.isnan(Xc)
        if np.all(obs):
            W_new = np.linalg.solve(
                np.sum(Ezz, axis=2),
                Ez @ Xc.T,
            ).T
            sig2_new = (
                np.mean(np.square(Xc))
                - np.mean(2 * Xc * (W_new @ Ez))
                + np.sum(Ezz * (W_new.T @ W_new)[:, :, None])
                / (self.n_features_in_ * self.n_samples_)
            )

        else:
            W_new = np.zeros((self.n_features_in_, self.n_components_))
            for i in range(self.n_features_in_):
                idx_obs = obs[i, :]
                W_new[i, :] = np.linalg.solve(
                    np.sum(Ezz[:, :, idx_obs], axis=2), Ez[:, idx_obs] @ Xc[i, idx_obs]
                )
            sig2_sum = 0
            for n in range(self.n_samples_):
                w_new = W_new[obs[:, n], :]
                sig2_sum += (
                    np.sum(np.square(Xc[obs[:, n], n]))
                    - 2 * Ez[:, n] @ w_new.T @ Xc[obs[:, n], n]
                    + np.sum(w_new.T * (Ezz[:, :, n] @ w_new.T))
                )
            sig2_new = sig2_sum / np.sum(obs)
        return W_new, sig2_new

    def _compute_nll(
        self,
        Xc: np.ndarray,
        W: np.ndarray,
        sig2: float,
    ):
        """Compute negative log likelihood under the current model.

        Parameters:

        Xc : (andarray of shape (n_samples, n_features))
            centered observations
        W : (ndarray of shape (n_features, n_components))
            transformation matrix
        sig2 : (float)
            variance parameter

        Returns:

        nll : (ndarray of shape (n_samples,))
            negative log likelihood of observations
        """
        obs = ~np.isnan(Xc)
        if np.all(obs):
            # use the matrix inversion lemma (because n_components < n_features)
            M = W.T @ W + sig2 * np.eye(self.n_components_)
            C_inv = (
                np.eye(self.n_features_in_) / sig2 - W @ np.linalg.solve(M, W.T) / sig2
            )
            nll = (
                self.n_features_in_ * np.log(2 * np.pi)
                - np.linalg.slogdet(C_inv)[1]
                + np.sum(Xc.T * (Xc.T @ C_inv), axis=1)
            ) / 2
        else:
            nll = np.zeros(self.n_samples_)
            for n in range(self.n_samples_):
                idx_obs = obs[:, n]
                x = Xc[idx_obs, n]
                w = W[idx_obs, :]
                # use the matrix inversion lemma (because n_components < n_features)
                m = w.T @ w + sig2 * np.eye(self.n_components_)
                c_inv = (
                    np.eye(np.sum(idx_obs)) / sig2 - w @ np.linalg.solve(m, w.T) / sig2
                )
                nll[n] = (
                    np.sum(idx_obs) * np.log(2 * np.pi)
                    - np.linalg.slogdet(c_inv)[1]
                    + np.sum(x * (x @ c_inv))
                ) / 2
        return nll
