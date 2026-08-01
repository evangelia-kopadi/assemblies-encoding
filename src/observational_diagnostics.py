"""Observational diagnostics to illustrate correlation is not causation.

Full method name: observational association diagnostics with correlation, partial
correlation, and feature-variance checks.

How it works: binarize or numeric-convert observed variables; compute Pearson and
partial-correlation matrices; rank highly correlated non-edges against the
ground-truth graph; report near-constant features that can weaken PC/GES; and
provide context for interpreting observational discovery results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


Edge = Tuple[str, str]
UndirectedEdge = Tuple[str, str]


def _as_undirected(edge: Edge) -> UndirectedEdge:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def ground_truth_undirected_adjacency(ground_truth_edges: Sequence[Edge]) -> Set[UndirectedEdge]:
    return {_as_undirected(e) for e in ground_truth_edges}


def binarize_dataframe(
    df: pd.DataFrame,
    var_names: Sequence[str],
    *,
    positive_values_map: Optional[Dict[str, Set[object]]] = None,
) -> pd.DataFrame:
    """Convert a mixed/categorical dataframe into a numeric 0/1 matrix.

    If positive_values_map is provided, each variable is encoded as:
      1 if df[var] in positive_values_map[var], else 0

    If not provided, tries a best-effort numeric conversion.
    """

    out: Dict[str, np.ndarray] = {}
    for v in var_names:
        if positive_values_map is not None and v in positive_values_map:
            positives = positive_values_map[v]
            out[v] = df[v].isin(list(positives)).astype(float).to_numpy()
        else:
            # Best-effort numeric conversion; non-convertible -> NaN -> filled with median.
            series = pd.to_numeric(df[v], errors="coerce")
            if series.isna().all():
                # If nothing is numeric, fall back to categorical codes (still "observational").
                out[v] = df[v].astype("category").cat.codes.astype(float).to_numpy()
            else:
                filled = series.fillna(series.median())
                out[v] = filled.astype(float).to_numpy()

    return pd.DataFrame(out)


def correlation_matrix(data_df: pd.DataFrame, var_names: Sequence[str]) -> pd.DataFrame:
    """Pearson correlation matrix (works fine for binary 0/1 as phi correlation)."""

    x = data_df[list(var_names)].to_numpy(dtype=float)
    # Guard: constant columns -> NaNs in corrcoef.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=list(var_names), columns=list(var_names))


def partial_correlation_matrix(data_df: pd.DataFrame, var_names: Sequence[str]) -> pd.DataFrame:
    """Partial correlation via precision matrix (inverse covariance).

    For standardized data, partial corr between i,j is:
      -P_ij / sqrt(P_ii * P_jj)
    where P is the precision matrix.

    Uses pseudo-inverse for stability when covariance is singular.
    """

    x = data_df[list(var_names)].to_numpy(dtype=float)

    # Standardize (avoid zero std)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    z = (x - mu) / sigma

    cov = np.cov(z, rowvar=False)
    # Pseudo-inverse to handle collinearity.
    precision = np.linalg.pinv(cov)

    p = precision
    denom = np.sqrt(np.outer(np.diag(p), np.diag(p)))
    with np.errstate(invalid="ignore", divide="ignore"):
        pcorr = -p / denom

    pcorr = np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(pcorr, 1.0)
    return pd.DataFrame(pcorr, index=list(var_names), columns=list(var_names))


@dataclass(frozen=True)
class EdgeAssociation:
    a: str
    b: str
    abs_corr: float
    corr: float
    abs_partial_corr: float
    partial_corr: float


def top_correlated_pairs(
    corr: pd.DataFrame,
    partial_corr: pd.DataFrame,
    *,
    k: int = 10,
    exclude_pairs: Optional[Set[UndirectedEdge]] = None,
) -> List[EdgeAssociation]:
    """Return top-k pairs by |corr|, with partial-corr side by side."""

    names = list(corr.columns)
    exclude_pairs = exclude_pairs or set()

    pairs: List[EdgeAssociation] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            u = (a, b) if a <= b else (b, a)
            if u in exclude_pairs:
                continue
            c = float(corr.loc[a, b])
            pc = float(partial_corr.loc[a, b])
            pairs.append(
                EdgeAssociation(
                    a=a,
                    b=b,
                    abs_corr=abs(c),
                    corr=c,
                    abs_partial_corr=abs(pc),
                    partial_corr=pc,
                )
            )

    pairs.sort(key=lambda p: p.abs_corr, reverse=True)
    return pairs[:k]


@dataclass(frozen=True)
class FeatureVarianceRow:
    name: str
    variance: float
    min_value: float
    max_value: float
    unique_count: int


def feature_variance_report(features_df: pd.DataFrame, var_names: Sequence[str]) -> List[FeatureVarianceRow]:
    rows: List[FeatureVarianceRow] = []
    for v in var_names:
        col = features_df[v].to_numpy(dtype=float)
        rows.append(
            FeatureVarianceRow(
                name=v,
                variance=float(np.var(col)),
                min_value=float(np.min(col)),
                max_value=float(np.max(col)),
                unique_count=int(pd.Series(col).nunique(dropna=True)),
            )
        )
    rows.sort(key=lambda r: r.variance)
    return rows


def format_edge_table(rows: Iterable[EdgeAssociation]) -> str:
    header = "pair | corr | partial_corr | |corr| | |partial_corr|"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.a}-{r.b} | {r.corr:+.3f} | {r.partial_corr:+.3f} | {r.abs_corr:.3f} | {r.abs_partial_corr:.3f}"
        )
    return "\n".join(lines)


def print_correlation_vs_causation_diagnostics(
    *,
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    var_names: Sequence[str],
    ground_truth_edges: Sequence[Edge],
    label: str,
    top_k: int = 8,
) -> Dict[str, object]:
    """Print the requested diagnostic views for a given representation."""

    gt_adj = ground_truth_undirected_adjacency(ground_truth_edges)

    corr = correlation_matrix(raw_df, var_names)
    pcorr = partial_correlation_matrix(raw_df, var_names)

    non_edge_top = top_correlated_pairs(corr, pcorr, k=top_k, exclude_pairs=gt_adj)

    # Ground-truth edges as pairs (for reference)
    gt_pairs = []
    for a, b in ground_truth_edges:
        c = float(corr.loc[a, b])
        pc = float(pcorr.loc[a, b])
        gt_pairs.append(
            EdgeAssociation(
                a=a,
                b=b,
                abs_corr=abs(c),
                corr=c,
                abs_partial_corr=abs(pc),
                partial_corr=pc,
            )
        )
    gt_pairs.sort(key=lambda r: r.abs_corr, reverse=True)

    # Feature variance (to spot saturation/constant features)
    var_rows = feature_variance_report(feature_df, var_names)

    print("\n" + "=" * 70)
    print(f"CORRELATION != CAUSATION DIAGNOSTICS ({label})")
    print("=" * 70)

    print("\n[1] Correlation matrix (observational):")
    print(corr.round(3).to_string())

    print("\n[2] Partial correlation matrix (controls for other variables):")
    print(pcorr.round(3).to_string())

    print(f"\n[3] Top {top_k} correlated NON-EDGES (high correlation but NOT causal edges in GT):")
    print(format_edge_table(non_edge_top))

    print("\n[4] Ground-truth edges with their (partial) correlations:")
    print(format_edge_table(gt_pairs))

    print("\n[5] Feature variance report (catch saturation/constant features):")
    for r in var_rows:
        flag = " (LOW VAR)" if r.variance < 1e-6 else ""
        print(f"  {r.name:20s} var={r.variance:.6f} min={r.min_value:.3f} max={r.max_value:.3f} uniq={r.unique_count}{flag}")

    return {
        "corr": corr,
        "partial_corr": pcorr,
        "top_correlated_non_edges": non_edge_top,
        "gt_edge_associations": gt_pairs,
        "feature_variance": var_rows,
    }



def print_observational_association_diagnostics(
    *,
    raw_df: pd.DataFrame,
    var_names: Sequence[str],
    ground_truth_edges: Sequence[Edge],
    label: str,
    top_k: int = 8,
) -> Dict[str, object]:
    """Print only the observational association views (corr + partial corr + edge-wise comparisons)."""

    gt_adj = ground_truth_undirected_adjacency(ground_truth_edges)
    corr = correlation_matrix(raw_df, var_names)
    pcorr = partial_correlation_matrix(raw_df, var_names)
    non_edge_top = top_correlated_pairs(corr, pcorr, k=top_k, exclude_pairs=gt_adj)

    gt_pairs = []
    for a, b in ground_truth_edges:
        c = float(corr.loc[a, b])
        pc = float(pcorr.loc[a, b])
        gt_pairs.append(
            EdgeAssociation(
                a=a,
                b=b,
                abs_corr=abs(c),
                corr=c,
                abs_partial_corr=abs(pc),
                partial_corr=pc,
            )
        )
    gt_pairs.sort(key=lambda r: r.abs_corr, reverse=True)

    print("\n" + "=" * 70)
    print(f"OBSERVATIONAL ASSOCIATIONS ({label})")
    print("=" * 70)

    print("\n[1] Correlation matrix (observational):")
    print(corr.round(3).to_string())

    print("\n[2] Partial correlation matrix (controls for other variables):")
    print(pcorr.round(3).to_string())

    print(f"\n[3] Top {top_k} correlated NON-EDGES (high correlation but NOT causal edges in GT):")
    print(format_edge_table(non_edge_top))

    print("\n[4] Ground-truth edges with their (partial) correlations:")
    print(format_edge_table(gt_pairs))

    return {
        "corr": corr,
        "partial_corr": pcorr,
        "top_correlated_non_edges": non_edge_top,
        "gt_edge_associations": gt_pairs,
    }


def print_feature_variance_only(
    *,
    feature_df: pd.DataFrame,
    var_names: Sequence[str],
    label: str,
) -> List[FeatureVarianceRow]:
    """Print only the feature variance / saturation checks."""

    var_rows = feature_variance_report(feature_df, var_names)

    print("\n" + "=" * 70)
    print(f"FEATURE VARIANCE REPORT ({label})")
    print("=" * 70)

    for r in var_rows:
        flag = " (LOW VAR)" if r.variance < 1e-6 else ""
        print(f"  {r.name:20s} var={r.variance:.6f} min={r.min_value:.3f} max={r.max_value:.3f} uniq={r.unique_count}{flag}")

    return var_rows
