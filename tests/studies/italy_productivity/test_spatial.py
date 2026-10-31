"""Spatial weights, Moran's I and eigenvector filtering on a synthetic 6x6 grid of two countries (offline)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from cifresignificative.studies.italy_productivity.analysis import spatial as sp


@pytest.fixture(scope="module")
def grid():
    """Country AA is a 6x3 block of 1 km squares, BB another 6x3 block far away; one extra BB island (BB99) touches nothing."""
    geos, geom = [], {}
    for c, x0 in (("AA", 0), ("BB", 100_000)):
        for i in range(6):
            for j in range(3):
                code = f"{c}{i}{j}"
                geos.append(code)
                geom[code] = box(x0 + i * 1000, j * 1000, x0 + (i + 1) * 1000, (j + 1) * 1000)
    geos.append("BB99")
    geom["BB99"] = box(100_000, 50_000, 100_500, 50_500)
    return geos, geom


def test_contiguity_queen_and_island_fallback(grid):
    geos, geom = grid
    A = sp.contiguity(geos, geom, tol=10)
    ix = {g: i for i, g in enumerate(geos)}
    assert A.sum(1)[ix["AA11"]] == 8 and A.sum(1)[ix["AA00"]] == 3  # interior cell: all eight queen neighbours; corner: three
    assert not A[ix["AA00"], ix["BB00"]]  # never across countries
    assert A[ix["BB99"]].sum() == 1 and A[ix["BB99"], ix["BB02"]]  # the island is linked to the nearest cell of its own country
    assert (A == A.T).all()


def test_morans_i_smooth_versus_random(grid):
    geos, geom = grid
    W = sp.row_standardise(sp.contiguity(geos, geom, tol=10))
    rng = np.random.default_rng(0)
    smooth = np.array([int(g[2]) + int(g[3]) if g != "BB99" else 4 for g in geos], float)  # a plane through each block
    assert sp.morans_i(smooth, W)[0] > 0.4 and sp.morans_i(smooth, W)[1] > 3
    assert abs(sp.morans_i(rng.standard_normal(len(geos)), W)[1]) < 3
    groups = np.array([g[:2] for g in geos])
    assert sp.morans_perm(smooth, W, groups, 200, rng)[1] < 0.05


def test_filter_removes_spatial_autocorrelation_and_keeps_noise(grid):
    geos, geom = grid
    nb = sp.Neighbours(geos, geom, tol=10)
    rng = np.random.default_rng(1)
    row_w = sp.row_standardise(nb.A)
    smooth = pd.Series(sp.smooth_noise(np.zeros(len(geos)), row_w, 0.9, rng), index=geos)
    noise = pd.Series(rng.standard_normal(len(geos)), index=geos)
    X = pd.DataFrame({"smooth": smooth, "noise": noise})
    X = X - X.groupby([g[:2] for g in X.index]).transform("mean")
    F, d = nb.filter_table(X, min_n=10)
    d = d.set_index("indicator")
    assert d.loc["smooth", "moran_z"] > 2 and d.loc["smooth", "moran_z_filtered"] < 1.645 and d.loc["smooth", "k_vectors"] >= 1
    assert d.loc["noise", "k_vectors"] <= 1  # nothing to filter in white noise (at most one vector by chance)
    assert F["smooth"].notna().all() and abs(F["smooth"].corr(X["smooth"])) < 0.99


def test_candidate_vectors_are_orthogonal_to_country_means(grid):
    geos, geom = grid
    W = sp.row_standardise(sp.contiguity(geos, geom, tol=10))
    groups = np.array([g[:2] for g in geos])
    E = sp.candidate_vectors(W, groups)
    assert E.shape[1] >= 1
    for c in ("AA", "BB"):
        assert np.allclose(E[groups == c].sum(0), 0, atol=1e-8)


@pytest.mark.skipif(not Path("data/processed/regional_nuts2021.parquet").exists(), reason="NUTS 2021 table not built")
def test_notebook_25_screens_on_the_nuts2021_geography(tmp_path):
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    r = ex.run_spatial_screens(out=tmp_path, n_perm=1000, n_sim=1, n_perm_null=200)
    c = r["sp6_screen_counts"].set_index("screen")
    assert (c.pairs_nuts2021 == c.pairs_filtered).all() and (c.columns_after_filter <= 1).all() and (c.significant_nuts2021 > 0).all()
    assert (c.both_nuts2021_and_filtered <= c[["significant_nuts2021", "significant_filtered"]].min(axis=1)).all()
    n = r["sp7_screen_null"]
    assert n[n["null"] == "shuffled within countries"].discoveries.max() <= 1  # a shuffled screen finds (almost) nothing, plain or filtered
    p = r["sp5_screen_pairs"]
    assert set(p.screen) == {"levels", "changes"} and p.rho.abs().le(1).all()
