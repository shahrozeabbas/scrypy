"""Tests for binomial deviance feature selection."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import scrypy


def test_dense_top_n():
    X = np.array([[10, 0, 5], [5, 10, 5], [0, 1, 9]], dtype=float)
    ad = AnnData(X=X)
    scrypy.highly_deviant_genes(ad, n_top_genes=2, inplace=True)
    assert ad.var['highly_variable'].sum() == 2
    assert ad.var['binomial_deviance'].notna().all()
    assert ad.uns['hvg']['flavor'] == 'binomial_deviance'


def test_sparse_matches_dense():
    X = np.array([[10, 0, 5], [5, 10, 5], [0, 1, 9]], dtype=float)
    ad_d = AnnData(X=X.copy())
    ad_s = AnnData(X=sparse.csr_matrix(X))
    scrypy.highly_deviant_genes(ad_d, n_top_genes=3, inplace=True)
    scrypy.highly_deviant_genes(ad_s, n_top_genes=3, inplace=True)
    np.testing.assert_allclose(
        ad_d.var['binomial_deviance'].to_numpy(),
        ad_s.var['binomial_deviance'].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_batch_key():
    X = np.array([[10, 0, 5], [5, 10, 5], [0, 1, 9]], dtype=float)
    ad = AnnData(X=X)
    ad.obs['b'] = ['a', 'a', 'b']
    scrypy.highly_deviant_genes(ad, n_top_genes=2, batch_key='b', inplace=True)
    assert ad.var['highly_variable'].sum() == 2


def test_inplace_false_subset():
    X = np.array([[10, 0, 5], [5, 10, 5]], dtype=float)
    ad = AnnData(X=X)
    df = scrypy.highly_deviant_genes(ad, n_top_genes=1, inplace=False, subset=True)
    assert len(df) == 1
    assert 'highly_variable' not in ad.var.columns


def test_zero_total_cell_warns():
    X = np.array([[1, 0], [0, 0], [0, 2]], dtype=float)
    ad = AnnData(X=X)
    with pytest.warns(UserWarning, match='zero total'):
        scrypy.highly_deviant_genes(ad, n_top_genes=2, inplace=True)


def test_n_top_exceeds_n_vars():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    ad = AnnData(X=X)
    with pytest.warns(UserWarning, match='n_top_genes'):
        scrypy.highly_deviant_genes(ad, n_top_genes=10, inplace=True)
    assert ad.var['highly_variable'].all()
