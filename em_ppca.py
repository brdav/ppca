import numba as nb
import numpy as np


class EMPPCA:
    """Probabilistic Principle Component Analysis (PPCA).
    PPCA is a probabilistic latent variable model, whose maximum likelihood
    solution corresponds to PCA. For an introduction to PPCA, see [1].

    This implementation uses the expectation-maximization (EM) algorithm to
    find maximum-likelihood estimates of the PPCA model parameters. This
    enables a principled handling of missing values in the dataset. In this
    implementation, we use p(X_obs, Z) as the complete-data likelihood,
    where X_obs denotes the non-missing entries in the data X, and Z
    denotes the latent variables (see [4]).

    Parameters:

    n_components : (int)
        Number of components to estimate. Has to be smaller than
        min(n_features, n_samples).
    max_iter : (int)
        Maximum number of iterations to run EM algorithm.
    min_iter : (int)
        Minimum number of iterations to run EM algorithm.
    tol_nll : (float)
        Convergence criterion for EM: relative change in complete-data
        negative log likelihood.
    tol_delta : (float)
        Convergence criterion for EM: relative change in estimated
        parameters (components and residual variance).
    verbose : (bool)
        Whether to print convergence info.
    random_state : (int)
        Fix a random seed to replicate EM initial conditions.

    Attributes:

    For ease of use, this class has the same attributes as
    `sklearn.decomposition.PCA`. Quoting from the sklearn docs:

    <<
    components_ : (ndarray of shape (n_components, n_features))
        Principal axes in feature space, representing the
        directions of maximum variance in the data.
        Equivalently, the right singular vectors of the
        centered input data, parallel to its eigenvectors.
        The components are sorted by decreasing
        explained_variance_.

    explained_variance_ : (ndarray of shape (n_components,))
        The amount of variance explained by each of the
        selected components. The variance estimation
        uses n_samples - 1 degrees of freedom.
        Equal to n_components largest eigenvalues of the
        covariance matrix of X.

    explained_variance_ratio_ : (ndarray of shape (n_components,))
        Percentage of variance explained by each
        of the selected components.

    singular_values_ : (ndarray of shape (n_components,))
        The singular values corresponding to each of the
        selected components. The singular values are equal
        to the 2-norms of the n_components variables in the
        lower-dimensional space.

    mean_ : (ndarray of shape (n_features,))
        Per-feature empirical mean, estimated from the training set.
        Equal to X.mean(axis=0).

    n_components_ : (int)
        The number of components.

    n_samples_ : (int)
        Number of samples in the training data.

    noise_variance_ : (float)
        The estimated noise covariance following the Probabilistic PCA model.
        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    n_features_in_ : (int)
        Number of features seen during fit.
    >>

    References:
    [1] Bishop, C. M., Pattern Recognition and Machine Learning. New York:
        Springer, 2006.
    [2] Tipping, M. E. and Bishop, C. M., Probabilistic Principal Component
        Analysis. Journal of the Royal Statistical Society: Series B
        (Statistical Methodology), 1999.
    [3] Sam Roweis. EM algorithms for PCA and SPCA. In Proceedings of the
        1997 conference on Advances in neural information processing
        systems 10 (NIPS '97), 1998, 626-632.
    [4] Alexander Ilin and Tapani Raiko. 2010. Practical Approaches to
        Principal Component Analysis in the Presence of Missing Values. J.
        Mach. Learn. Res. 11 (August 2010), 1957-2000.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 10000,
        min_iter: int = 20,
        tol_nll: float = 1e-6,
        tol_delta: float = 1e-6,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        if tol_delta <= 0:
            raise ValueError("tol_delta must be greater than zero")
        if tol_nll <= 0:
            raise ValueError("tol_nll must be greater than zero")

        self.n_components = n_components
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol_nll = tol_nll
        self.tol_delta = tol_delta
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.array):

        # Code expects X as (n_features, n_samples)
        X = X.T

        n_features, n_samples = X.shape
        max_rank = min(n_samples, n_features)
        if self.n_components > max_rank - 1:
            raise ValueError(
                "For PPCA, n_components (= {0}) has to be smaller than min(n_samples, n_features) (= {1}).".format(
                    self.n_components, max_rank
                )
            )

        is_nan = np.isnan(X)
        has_missing_values = np.any(is_nan)
        obs = ~is_nan
        if np.any(np.all(is_nan, axis=0)):
            raise ValueError(
                "X contains observations which are entirely NaN. Please remove those before running PPCA."
            )

        if has_missing_values:
            Z, W, mu, sig2 = self._em_missing(X, obs)
        else:
            if self.verbose:
                print(
                    "Found no missing values, could also use `sklearn.decomposition.PCA`."
                )
            Z, W, mu, sig2 = self._em_complete(X)

        # Reconstruct X from latent space
        WTW = W.T @ W
        X_hat = (
            W @ np.linalg.inv(WTW) @ (WTW + sig2 * np.eye(self.n_components)) @ Z
            + mu[:, np.newaxis]
        )

        if self.verbose:
            diff = X - X_hat
            diff[~obs] = 0
            rms_resid = np.linalg.norm(diff, "fro") / np.sqrt(np.sum(obs))
            print("Root mean square reconstruction error = {}.".format(rms_resid))

        # Rotate W to the standard PCA basis
        W = np.linalg.svd(W, full_matrices=False)[0]
        scores = (X_hat.T - mu[np.newaxis, :]) @ W
        latent = np.linalg.eig(scores.T @ scores)[0]
        latent = np.sort(latent)[::-1] / (n_samples - 1)

        # The largest element in each principle component will have a positive sign.
        max_abs_u_cols = np.argmax(np.abs(W), axis=0)
        shift = np.arange(W.shape[1])
        indices = max_abs_u_cols + shift * W.shape[0]
        signs = np.sign(np.take(np.reshape(W.T, (-1,)), indices, axis=0))
        W *= signs[np.newaxis, :]
        scores *= signs[np.newaxis, :]

        small_latent = latent < (max(X.shape) * np.spacing(np.max(latent)))
        if np.any(small_latent):
            raise RuntimeError(
                "Data covariance matrix is close to singular, {0} of {1} eigenvalues are small.".format(
                    np.sum(small_latent), len(latent)
                )
            )

        self.transformed_values_ = scores

        self.components_ = W.T
        self.explained_variance_ = latent
        self.explained_variance_ratio_ = latent / (
            np.sum(latent) + sig2 * (n_features - self.n_components)
        )
        self.singular_values_ = np.sqrt(latent * (n_samples - 1))
        self.mean_ = mu
        self.n_components_ = self.n_components
        self.n_samples_ = n_samples
        self.noise_variance_ = sig2
        self.n_features_in_ = n_features

        return self

    def fit_transform(self, X: np.array):
        return self.fit(X).transformed_values_

    def _em_complete(self, X: np.array):
        n_features, n_samples = X.shape

        W = np.random.randn(n_features, self.n_components)
        mu = np.mean(X, axis=1)
        sig2 = np.random.randn()
        nll = np.inf

        X -= mu[:, np.newaxis]

        if self.verbose:
            print(
                "{0: >11} {1: >11} {2: >13} {3: >27}".format(
                    "Iteration", "Sigma^2", "|Delta W|", "Negative log-likelihood"
                )
            )
            strfmt = "{0: >11d} {1: >12f} {2: >13f} {3: >27f}"

        itercount = 0
        traceS = np.sum(X**2) / (n_samples - 1)

        while itercount < self.max_iter:
            itercount += 1

            SW = X @ (X.T @ W) / (n_samples - 1)
            M_inv = np.linalg.inv(W.T @ W + sig2 * np.eye(self.n_components))

            W_new = SW @ np.linalg.inv(
                sig2 * np.eye(self.n_components) + M_inv @ W.T @ SW
            )
            sig2_new = (traceS - np.trace(SW @ M_inv @ W_new.T)) / n_features

            dW = np.max(
                np.abs(W - W_new) / (np.sqrt(np.finfo(float).eps) + np.max(np.abs(W)))
            )
            dsig2 = np.abs(sig2 - sig2_new) / (np.finfo(float).eps + abs(sig2))
            delta = max(dW, dsig2)

            nll_new = self._compute_nll(
                X, W_new, sig2_new, n_samples, n_features, self.n_components
            )

            W = W_new
            sig2 = sig2_new

            if self.verbose and (itercount % 20 == 0):
                print(strfmt.format(itercount, sig2, dW, nll_new))

            if itercount > self.min_iter:
                if delta < self.tol_delta:
                    break
                elif (nll - nll_new) < self.tol_nll:
                    break

            nll = nll_new

            if itercount == self.max_iter:
                raise RuntimeError(
                    "Maximum number of iterations reached without convergence: {}".format(
                        self.max_iter
                    )
                )

        Z = M_inv @ W.T @ X

        return Z, W, mu, sig2

    def _em_missing(self, X: np.array, obs: np.array):
        n_features, n_samples = X.shape

        W = np.random.randn(n_features, self.n_components)
        mu = np.zeros(n_features)
        sig2 = np.random.randn()
        nll = np.inf

        if self.verbose:
            print(
                "{0: >11} {1: >11} {2: >13} {3: >27}".format(
                    "Iteration", "Sigma^2", "|Delta W|", "Negative log-likelihood"
                )
            )
            strfmt = "{0: >11d} {1: >12f} {2: >13f} {3: >27f}"

        itercount = 0
        while itercount < self.max_iter:
            itercount += 1

            Z, C = self._e_step_missing(
                X, W, mu, sig2, obs, n_samples, self.n_components
            )

            W_new, mu_new, sig2_new = self._m_step_missing(
                X,
                Z,
                C,
                W,
                sig2,
                obs,
                n_samples,
                n_features,
                self.n_components,
            )

            nll_new = self._compute_nll_missing(
                X, W_new, mu_new, sig2_new, obs, n_samples, self.n_components
            )

            dW = np.max(
                np.abs(W - W_new) / (np.sqrt(np.finfo(float).eps) + np.max(np.abs(W)))
            )
            dsig2 = np.abs(sig2 - sig2_new) / (np.finfo(float).eps + sig2)
            delta = max(dW, dsig2)

            W = W_new.copy()
            mu = mu_new.copy()
            sig2 = sig2_new

            if itercount > self.min_iter:
                if delta < self.tol_delta:
                    break
                elif (nll - nll_new) < self.tol_nll:
                    break

            nll = nll_new

            if self.verbose and (itercount % 20 == 0):
                print(strfmt.format(itercount, sig2, dW, nll))

            if itercount == self.max_iter:
                raise RuntimeError(
                    "Maximum number of iterations reached without convergence: {}".format(
                        self.max_iter
                    )
                )

        mu_Z = np.mean(Z, axis=1)
        Z -= mu_Z[:, np.newaxis]
        mu += W @ mu_Z

        return Z, W, mu, sig2

    @staticmethod
    def _compute_nll(
        X: np.array,
        W: np.array,
        sig2: float,
        n_samples: int,
        n_features: int,
        n_components: int,
    ):
        # Use the matrix inversion lemma (because n_components < n_features)
        CC = (
            np.eye(n_features) / sig2
            - W @ np.linalg.solve(np.eye(n_components) + W.T @ W / sig2, W.T) / sig2**2
        )
        CC = W @ W.T + sig2 * np.eye(n_features)
        nll = (
            (
                n_features * np.log(2 * np.pi)
                - np.linalg.slogdet(CC)[1]
                + np.sum(X.T * (X.T @ CC)) / (n_samples - 1)
            )
            * n_samples
            / 2
        )
        return nll

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def _e_step_missing(
        X: np.array,
        W: np.array,
        mu: np.array,
        sig2: float,
        obs: np.array,
        n_samples: int,
        n_components: int,
    ):
        Z = np.zeros((n_components, n_samples))
        C = np.zeros((n_components, n_components, n_samples))
        for j in nb.prange(n_samples):
            x = X[:, j]
            idx_obs = obs[:, j]
            w = W[idx_obs, :]
            Cj = np.linalg.inv(sig2 * np.eye(n_components) + w.T @ w)
            Z[:, j] = Cj @ w.T @ (x[idx_obs] - mu[idx_obs])
            C[:, :, j] = Cj
        return Z, C

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def _m_step_missing(
        X: np.array,
        Z: np.array,
        C: np.array,
        W: np.array,
        sig2: float,
        obs: np.array,
        n_samples: int,
        n_features: int,
        n_components: int,
    ):
        mu_new = np.empty(n_features)
        W_new = np.zeros((n_features, n_components))

        resid = X - W @ Z
        for i in nb.prange(n_features):
            mu_new[i] = np.nanmean(resid[i, :])
            idx_obs = obs[i, :]
            M = Z[:, idx_obs] @ Z[:, idx_obs].T + sig2 * np.sum(
                C[:, :, idx_obs], axis=2
            )
            ww = Z[:, idx_obs] @ (X[i, idx_obs] - mu_new[i])
            W_new[i, :] = np.linalg.solve(M, ww)

        sig2_sum = 0
        for j in nb.prange(n_samples):
            w_new = W_new[obs[:, j], :]
            sig2_sum += np.sum(
                (
                    X[obs[:, j], j]
                    - w_new @ np.ascontiguousarray(Z[:, j])
                    - mu_new[obs[:, j]]
                )
                ** 2
            ) + sig2 * np.sum(w_new * (w_new @ np.ascontiguousarray(C[:, :, j])))
        sig2_new = sig2_sum / np.sum(obs)

        return W_new, mu_new, sig2_new

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def _compute_nll_missing(
        X: np.array,
        W: np.array,
        mu: np.array,
        sig2: float,
        obs: np.array,
        n_samples: int,
        n_components: int,
    ):
        nll = 0
        for j in nb.prange(n_samples):
            idx_obs = obs[:, j]
            x = X[idx_obs, j] - mu[idx_obs]
            W_obs = W[idx_obs, :]
            # Use the matrix inversion lemma (because n_components < n_features)
            Cy = (
                np.eye(np.sum(idx_obs)) / sig2
                - W_obs
                @ np.linalg.solve(
                    np.eye(n_components) + W_obs.T @ W_obs / sig2, W_obs.T
                )
                / sig2**2
            )
            nll += (
                np.sum(idx_obs) * np.log(2.0 * np.pi)
                - np.linalg.slogdet(Cy)[1]
                + np.sum(x * (x @ Cy))
            ) / 2
        return nll
