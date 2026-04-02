# scrypy

Python implementation of binomial deviance feature selection for single-cell RNA-Seq, based on the multinomial model from [Townes et al. (2019)](https://doi.org/10.1186/s13059-019-1861-6) and the [`scry`](https://bioconductor.org/packages/scry.html) R package.

## Installation

```bash
pip install scrypy
```

## Usage

```python
import scrypy

scrypy.highly_deviant_genes(adata, n_top_genes=2000)
```

Expects raw (or raw-like) non-negative integer UMI counts in `adata.X` or a named layer.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Annotated data matrix (cells x genes) |
| `n_top_genes` | `int` | Number of top genes to select |
| `layer` | `str \| None` | Layer to use instead of `adata.X` |
| `subset` | `bool` | Subset `adata` to selected genes (default `False`) |
| `inplace` | `bool` | Write results into `adata.var` (default `True`) |
| `batch_key` | `str \| None` | Obs key for batch; deviance is summed across batches |
| `check_values` | `bool` | Warn if data are not non-negative integers (default `True`) |

### Outputs

When `inplace=True`:

- `adata.var['binomial_deviance']` -- deviance score per gene
- `adata.var['highly_variable']` -- boolean selection mask

## Citation

If you use `scrypy`, please cite the original method paper:

> Townes FW, Hicks SC, Aryee MJ, Irizarry RA (2019).
> **Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model.**
> *Genome Biology* 20:295.
> [https://doi.org/10.1186/s13059-019-1861-6](https://doi.org/10.1186/s13059-019-1861-6)

If citing this Python implementation specifically:

> scrypy: Binomial deviance feature selection for AnnData (2026).
