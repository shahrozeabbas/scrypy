"""Binomial deviance feature selection (Townes et al. 2019; matches scry R implementation)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.special import xlogy

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix, csr_matrix

_BINO_FLAVOR = 'binomial_deviance'


def _check_nonnegative_integers(x: np.ndarray | csr_matrix | csc_matrix) -> bool:
    """Lightweight count check without a scanpy dependency."""
    if sparse.issparse(x):
        data = x.data
        if data.size == 0:
            return True
        return bool(np.all(np.mod(data, 1) == 0) and np.all(data >= 0))

    if not np.issubdtype(x.dtype, np.integer):
        if not np.all(np.mod(x, 1) == 0):
            return False
    return bool(np.all(x >= 0))


def _get_x(adata: AnnData, layer: str | None) -> np.ndarray | csr_matrix | csc_matrix:
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        msg = f"layer {layer!r} not found in adata.layers"
        raise KeyError(msg)
    return adata.layers[layer]


def _row_sums(x: np.ndarray | csr_matrix | csc_matrix) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.sum(axis=1)).ravel()
    return np.asarray(x.sum(axis=1), dtype=np.float64).ravel()


def _filter_zero_sum_cells(
    x: np.ndarray | csr_matrix | csc_matrix, sz: np.ndarray
) -> tuple[np.ndarray | csr_matrix | csc_matrix, np.ndarray, np.ndarray]:
    """Drop rows with zero total count; warn if any."""
    mask = sz > 0
    if not np.all(mask):
        n_bad = int(np.sum(~mask))
        warnings.warn(
            f'excluding {n_bad} cell(s) with zero total counts from deviance',
            UserWarning,
            stacklevel=3,
        )
        if sparse.issparse(x):
            x_f = x[mask]
        else:
            x_f = np.asarray(x[mask], dtype=np.float64)
        sz_f = sz[mask]
        return x_f, sz_f, mask
    return x, sz, mask


def _sparse_csc_binomial_deviance(x_csc: csc_matrix, sz: np.ndarray) -> np.ndarray:
    """Gene-wise binomial deviance; x is (n_obs, n_vars) CSC; sz is row sums.

    Fully vectorized over nonzero entries using np.bincount.
    Memory is O(nnz + n_obs + n_vars), never O(n_obs * n_vars).
    """
    n_obs, n_vars = x_csc.shape
    assert sz.shape == (n_obs,)
    feature_sums = np.asarray(x_csc.sum(axis=0), dtype=np.float64).ravel()
    sz_sum = float(np.sum(sz))
    if sz_sum <= 0:
        return np.zeros(n_vars, dtype=np.float64)

    indptr = x_csc.indptr
    data = x_csc.data.astype(np.float64, copy=False)
    row_idx = x_csc.indices
    col_idx = np.repeat(np.arange(n_vars, dtype=np.intp), np.diff(indptr))

    sz_nz = sz[row_idx]
    ratio = data / sz_nz

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(ratio)
        l1p = np.log1p(-ratio)

    nz_contrib = data * (log_ratio - l1p) + sz_nz * l1p
    nz_contrib = np.nan_to_num(nz_contrib, nan=0.0, posinf=0.0, neginf=0.0)

    ll_sat = np.bincount(col_idx, weights=nz_contrib, minlength=n_vars)

    p = feature_sums / sz_sum
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
    l1p_null = np.log1p(-p_safe)
    ll_null = feature_sums * (np.log(p_safe) - l1p_null) + sz_sum * l1p_null
    ll_null[feature_sums <= 0] = 0.0

    dev = 2.0 * (ll_sat - ll_null)
    dev[~np.isfinite(dev)] = 0.0
    dev[feature_sums <= 0] = 0.0
    return dev


def _dense_binomial_deviance(x: np.ndarray, sz: np.ndarray) -> np.ndarray:
    """Gene-wise binomial deviance using scipy.special.xlogy (stable)."""
    x = np.asarray(x, dtype=np.float64)
    n_obs, n_vars = x.shape
    sz = np.asarray(sz, dtype=np.float64).ravel()
    if sz.shape[0] != n_obs:
        msg = 'sz length must match number of rows of X'
        raise ValueError(msg)

    feature_sums = np.sum(x, axis=0)
    sz_sum = float(np.sum(sz))
    if sz_sum <= 0:
        return np.zeros(n_vars, dtype=np.float64)

    p = feature_sums / sz_sum
    eps = 1e-15
    p_safe = np.clip(p, eps, 1.0 - eps)

    sz_col = sz[:, np.newaxis]
    exp1 = sz_col * p_safe
    exp2 = sz_col * (1.0 - p_safe)

    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = xlogy(x, x / exp1)
        term2 = xlogy(sz_col - x, (sz_col - x) / exp2)

    dev = 2.0 * np.sum(term1 + term2, axis=0)
    dev[~np.isfinite(dev)] = 0.0
    dev[feature_sums <= 0] = 0.0
    return dev.astype(np.float64, copy=False)


def _binomial_deviance_gene_wise(x: np.ndarray | csr_matrix | csc_matrix, sz: np.ndarray) -> np.ndarray:
    """Compute per-gene binomial deviance for X (n_obs x n_vars)."""
    if sparse.issparse(x):
        x_csc = x.tocsc()
        return _sparse_csc_binomial_deviance(x_csc, sz)
    return _dense_binomial_deviance(x, sz)


def _compute_deviance_batch(
    x: np.ndarray | csr_matrix | csc_matrix,
    batch_labels: np.ndarray,
) -> np.ndarray:
    """Sum gene-wise binomial deviance within each batch."""
    if x.shape[0] != len(batch_labels):
        msg = 'batch_key length must match number of observations'
        raise ValueError(msg)

    unique = np.unique(batch_labels)
    total = None
    for b in unique:
        m = batch_labels == b
        if not np.any(m):
            continue
        x_b = x[m] if not sparse.issparse(x) else x[m]
        sz_b = _row_sums(x_b)
        x_b, sz_b, _ = _filter_zero_sum_cells(x_b, sz_b)
        if x_b.shape[0] == 0:
            continue
        d_b = _binomial_deviance_gene_wise(x_b, sz_b)
        total = d_b if total is None else total + d_b

    if total is None:
        return np.zeros(x.shape[1], dtype=np.float64)
    return total


def highly_deviant_genes(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int,
    subset: bool = False,
    inplace: bool = True,
    batch_key: str | None = None,
    check_values: bool = True,
) -> pd.DataFrame | None:
    """Select highly deviant genes via binomial deviance (multinomial null).

    Writes ``adata.var['binomial_deviance']``, ``adata.var['highly_variable']``,
    and ``adata.uns['hvg']`` when ``inplace=True``.

    Parameters
    ----------
    adata
        Annotated data matrix (cells × genes); expects raw-like counts.
    layer
        If set, use ``adata.layers[layer]`` instead of ``adata.X``.
    n_top_genes
        Number of top genes to mark as ``highly_variable``.
    subset
        If True, subset ``adata`` to highly deviant genes (inplace path only).
    inplace
        If False, return a DataFrame and do not modify ``adata``.
    batch_key
        Observation key; deviance is computed within each batch and summed.
    check_values
        If True, warn when values are not non-negative integers.

    Returns
    -------
    None if ``inplace=True``, else a DataFrame indexed by gene names with
    columns ``binomial_deviance`` and ``highly_variable``.
    """
    x = _get_x(adata, layer)
    computed_on = layer if layer is not None else 'adata.X'

    if check_values and not _check_nonnegative_integers(x):
        warnings.warn(
            'expected non-negative integer count data',
            UserWarning,
            stacklevel=2,
        )

    if batch_key is not None:
        if batch_key not in adata.obs:
            msg = f'batch_key {batch_key!r} not found in adata.obs'
            raise KeyError(msg)
        bl = np.asarray(adata.obs[batch_key])
        dev = _compute_deviance_batch(x, bl)
    else:
        sz = _row_sums(x)
        x_f, sz_f, _ = _filter_zero_sum_cells(x, sz)
        dev = _binomial_deviance_gene_wise(x_f, sz_f)

    n_vars = adata.shape[1]
    if dev.shape[0] != n_vars:
        msg = 'deviance length must match adata.n_vars'
        raise ValueError(msg)

    dev = np.nan_to_num(dev, nan=0.0, posinf=0.0, neginf=0.0)

    n_top = int(n_top_genes)
    if n_top > n_vars:
        warnings.warn(
            f'n_top_genes ({n_top}) > n_vars ({n_vars}); marking all genes',
            UserWarning,
            stacklevel=2,
        )
        n_top = n_vars

    order = np.argsort(-dev)
    highly_variable = np.zeros(n_vars, dtype=bool)
    if n_top > 0:
        highly_variable[order[:n_top]] = True

    df = pd.DataFrame(
        {
            'binomial_deviance': dev,
            'highly_variable': highly_variable,
        },
        index=adata.var_names.copy(),
    )

    if not inplace:
        out = df
        if subset:
            out = out.loc[out['highly_variable']]
        return out

    adata.var['binomial_deviance'] = df['binomial_deviance'].to_numpy()
    adata.var['highly_variable'] = df['highly_variable'].to_numpy()
    adata.uns['hvg'] = {'flavor': _BINO_FLAVOR, 'computed_on': computed_on}

    if subset:
        adata._inplace_subset_var(df['highly_variable'].to_numpy())

    return None
