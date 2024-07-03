# Probabilistic PCA (PPCA) in Python

## Principled approach for PCA with missing values via the EM algorithm

PPCA is a probabilistic latent variable model, whose maximum likelihood solution corresponds to PCA. For an introduction to PPCA, see [1].

This implementation uses the expectation-maximization (EM) algorithm to find maximum-likelihood estimates of the PPCA model parameters. This enables a principled handling of missing values in the dataset. In this implementation, we use `p(X_obs, Z)` as the complete-data likelihood, where `X_obs` denotes the non-missing entries in the data `X`, and `Z` denotes the latent variables (see [4]).

This implementation was tested for correctness against the [MATLAB ppca function](https://mathworks.com/help/stats/ppca.html).

## Requirements

This implementation uses `numba` (in addition to `numpy`) to accelerate the EM algorithm in the presence of missing values. It was tested with Python 3.12 and can be installed as a package with:
```
pip install -e .
```

To run the `demo.ipynb` notebook, the following packages are additionally required:
```
pip install notebook
pip install scikit-learn
pip install matplotlib
```

## Demo

<img src="./demo.png" width="900"/>

In the `demo.ipynb` we show basic usage and compare this implementation to the `sklearn` implementation. In short, PPCA can be used exactly like its `sklearn` counterpart:
```
from em_ppca import EMPPCA
...
em_ppca = EMPPCA(n_components=3)
# X contains data with possibly missing values (= np.nan)
# Z are the transformed values
Z = em_ppca.fit_transform(X)
print("principle components: ", em_ppca.components_)
...
```

## Limitations

The current implementation supports all attributes, but only the `fit` and `fit_transform` methods compared to the `sklearn` counterpart. Feedback / contributions are welcome!

## References

[1] Bishop, C. M., Pattern Recognition and Machine Learning. New York: Springer, 2006.

[2] Tipping, M. E. and Bishop, C. M., Probabilistic Principal Component Analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 1999.

[3] Sam Roweis. EM algorithms for PCA and SPCA. In Proceedings of the 1997 conference on Advances in neural information processing systems 10 (NIPS '97), 1998, 626-632.

[4] Alexander Ilin and Tapani Raiko. 2010. Practical Approaches to Principal Component Analysis in the Presence of Missing Values. J. Mach. Learn. Res. 11 (August 2010), 1957-2000.