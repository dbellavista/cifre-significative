"""Latent structure on synthetic data: planted factors are retained, pure noise is not, loadings are stable."""
import numpy as np
import pandas as pd

from cifresignificative.analysis import latent as L


def _Z(n=40, p=30, k=2, noise=0.5, seed=2):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, k))
    W = rng.normal(size=(k, p))
    A = F @ W + rng.normal(scale=noise, size=(n, p)) * np.sqrt(k)
    Z = pd.DataFrame(A, index=[f"C{i:02d}" for i in range(n)], columns=[f"x{j}" for j in range(p)])
    return (Z - Z.mean()) / Z.std(ddof=0)


def test_parallel_analysis_retains_planted_factors_and_none_in_noise():
    assert L.parallel_analysis(_Z(), n_perm=50)["n_retain"] == 2
    rng = np.random.default_rng(3)
    N = pd.DataFrame(rng.normal(size=(40, 30)))
    assert L.parallel_analysis((N - N.mean()) / N.std(ddof=0), n_perm=50)["n_retain"] == 0


def test_loadings_are_stable_and_signed():
    Z = _Z(n=60)
    st = L.bootstrap_stability(Z, 2, n_boot=40)
    assert (st.median_cosine > 0.9).all()
    V = L.pca(Z, 2)["loadings"]
    assert all(V[c].iloc[V[c].abs().argmax()] > 0 for c in V)
    assert abs(L.pca(Z)["explained"].sum() - 1) < 1e-9


def test_standardise_winsorises_and_fills_missing():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, np.nan]})
    Z = L.standardise(X, X.mean(), X.std(ddof=0))
    assert Z.a.max() <= 3.0 and Z.a.iloc[-1] == 0.0
