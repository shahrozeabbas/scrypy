# scrypy

Python implementation of binomial deviance feature selection for **single-cell count data** (non-negative integer matrices), following the multinomial-model view in [Townes et al. (2019)](https://doi.org/10.1186/s13059-019-1861-6) and the [`scry`](https://bioconductor.org/packages/scry.html) R package. Typical uses include scRNA-seq UMIs and scATAC-seq (or similar) peak or bin counts in `AnnData`; the method ranks **features** (`adata.var`) by deviance under a common null proportion.

## Installation

```bash
pip install scrypy
```

## Usage

```python
import scrypy

scrypy.highly_deviant_features(adata, n_top_features=2000)
```

Expects raw (or raw-like) **non-negative integer counts** in `adata.X` or a named layer (e.g. UMIs, or ATAC fragments per peak).

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Count matrix (cells × features; e.g. genes or peaks) |
| `n_top_features` | `int` | Number of top features to select |
| `layer` | `str \| None` | Layer to use instead of `adata.X` |
| `subset` | `bool` | Subset `adata` to selected features (default `False`) |
| `inplace` | `bool` | Write results into `adata.var` (default `True`) |
| `batch_key` | `str \| None` | Obs key for batch; deviance is summed across batches |
| `check_values` | `bool` | Warn if data are not non-negative integers (default `True`) |

### Outputs

When `inplace=True`:

- `adata.var['binomial_deviance']` -- deviance score per feature
- `adata.var['highly_variable']` -- boolean selection mask

## Citation

If you use `scrypy`, cite the original method paper (written in the scRNA-seq setting; the deviance feature-screening idea applies to other **multinomial-style count tables** as well):

> Townes FW, Hicks SC, Aryee MJ, Irizarry RA (2019).
> **Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model.**
> *Genome Biology* 20:295.
> [https://doi.org/10.1186/s13059-019-1861-6](https://doi.org/10.1186/s13059-019-1861-6)

If citing this Python implementation specifically:

> scrypy: Binomial deviance feature selection for AnnData (2026).
