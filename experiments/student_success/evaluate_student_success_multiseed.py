"""Multi-seed evaluation for Student Success (3-DAG + do() effects).

Runs many random seeds to move from a single-run demo to aggregate statistics.

Outputs (all under runs/YYYYMMDD):
- summary.csv
- dodeltas.csv
- report.md
"""

from __future__ import annotations

import argparse
import json
import datetime
import os
import sys
from dataclasses import asdict

# Ensure repo root is on sys.path so "import src.*" and "import experiments.*" resolve locally
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from experiments.experiment_defaults import DEFAULTS, runner_kwargs, _cfg, get_output_filepath

_ms = _cfg.get("multiseed", {})
from src.runner import run_causal_dag_validation
from src.encoding.bernoulli import encode_bernoulli_dataframe
from src.representation.assembly_feature_extraction import extract_assembly_features

# Reuse the SCM + do() generator from the case study implementation without
# requiring experiments/ to be a Python package.
import importlib.util

_case_study_path = os.path.join(os.path.dirname(__file__), 'validate_student_success.py')
_spec = importlib.util.spec_from_file_location('validate_student_success', _case_study_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f'Failed to load case study module at {_case_study_path}')
_case_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_case_study)

generate_student_data = _case_study.generate_student_data


VAR_NAMES = [
    'PriorGPA',
    'Attendance',
    'StudyHours',
    'FinalExamScore',
    'FinalGrade',
]

POSITIVE_VALUES_MAP = {
    'PriorGPA': {'High'},
    'Attendance': {'Good'},
    'StudyHours': {'High'},
    'FinalExamScore': {'Excellent'},
    'FinalGrade': {'Excellent'},
}

INTERVENTIONS: list[tuple[str, dict[str, str]]] = [
    ('do(Attendance=Good)', {'Attendance': 'Good'}),
    ('do(StudyHours=High)', {'StudyHours': 'High'}),
    ('do(PriorGPA=High)', {'PriorGPA': 'High'}),
    ('do(FinalExamScore=Excellent)', {'FinalExamScore': 'Excellent'}),
]


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation using pandas ranking (handles ties)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float('nan')
    rx = pd.Series(x).rank(method='average').to_numpy(dtype=float)
    ry = pd.Series(y).rank(method='average').to_numpy(dtype=float)
    c = np.corrcoef(rx, ry)
    return float(c[0, 1])


def _ci95_mean(x: np.ndarray) -> tuple[float, float, float]:
    """Normal-approx 95% CI for the mean (good for n~50+)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * (sd / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return mu, mu - half, mu + half


def _ols_slope(x: np.ndarray, y: np.ndarray, *, fit_intercept: bool = False) -> dict:
    """Simple least-squares calibration y  a*x (+ b). Returns a,b,r2,mae."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return {'n': int(len(x)), 'a': float('nan'), 'b': float('nan'), 'r2': float('nan'), 'mae': float('nan')}

    if fit_intercept:
        X = np.column_stack([x, np.ones_like(x)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0])
        b = float(beta[1])
        yhat = a * x + b
    else:
        denom = float(np.dot(x, x))
        a = float(np.dot(x, y) / denom) if denom else float('nan')
        b = 0.0
        yhat = a * x

    resid = y - yhat
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else float('nan')
    mae = float(np.mean(np.abs(resid)))
    return {'n': int(len(x)), 'a': a, 'b': b, 'r2': r2, 'mae': mae}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float('nan')
    c = np.corrcoef(x, y)
    return float(c[0, 1])



def _df_markdown_table(df: pd.DataFrame, *, float_digits: int = 3) -> str:
    cols = list(df.columns)
    def _fmt(v):
        if v is None:
            return ''
        try:
            import math
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return ''
        except Exception:
            pass
        if isinstance(v, float):
            return f'{v:.{float_digits}f}'
        return str(v)

    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows = []
    for _, row in df.iterrows():
        rows.append('| ' + ' | '.join(_fmt(row[c]) for c in cols) + ' |')
    return '\n'.join([header, sep] + rows)


def _p_excellent(df: pd.DataFrame, col: str) -> float:
    return float((df[col] == 'Excellent').mean())


def _extract_variable_features_from_neural(neural: np.ndarray, *, neurons_per_var: int) -> pd.DataFrame:
    feats: dict[str, np.ndarray] = {}
    for var_idx, var in enumerate(VAR_NAMES):
        start = var_idx * neurons_per_var
        end = start + neurons_per_var
        acts = neural[:, start:end]
        feats[var] = acts.mean(axis=1)
    return pd.DataFrame(feats)


def _features_for_df(
    df: pd.DataFrame,
    *,
    brain: object,
    encoding_seed: int,
    neurons_per_var: int,
    positive_prob: float,
    negative_prob: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neural = encode_bernoulli_dataframe(
        df,
        list(VAR_NAMES),
        neurons_per_var=neurons_per_var,
        positive_values_map=POSITIVE_VALUES_MAP,
        positive_prob=positive_prob,
        negative_prob=negative_prob,
        seed=encoding_seed,
    )

    neuron_df = _extract_variable_features_from_neural(neural, neurons_per_var=neurons_per_var)

    source_var_by_target_area_name = {v: v for v in VAR_NAMES}
    assembly_features = extract_assembly_features(
        neural,
        brain,
        list(VAR_NAMES),
        neurons_per_var,
        source_var_by_target_area_name=source_var_by_target_area_name,
    )
    assembly_features_base = {k: v for k, v in assembly_features.items() if '_x_' not in k}
    assembly_df = pd.DataFrame(assembly_features_base)

    return neuron_df, assembly_df


def run_seed(
    *,
    seed: int,
    n_samples: int,
    n_eval: int,
    alpha_primary: float,
    alpha_fallback: float,
) -> tuple[dict, list[dict]]:
    df, ground_truth = generate_student_data(n_students=n_samples, seed=seed, do=None)

    # 3-DAG validation
    used_alpha = alpha_primary
    results = run_causal_dag_validation(
        df=df,
        var_names=list(VAR_NAMES),
        ground_truth_edges=ground_truth,
        positive_values_map=POSITIVE_VALUES_MAP,
        **runner_kwargs(seed=seed, alpha_pc=alpha_primary),
    )

    if (len(results.get('neuron_edges', [])) == 0) or (len(results.get('assembly_edges', [])) == 0):
        used_alpha = alpha_fallback
        results = run_causal_dag_validation(
            df=df,
            var_names=list(VAR_NAMES),
            ground_truth_edges=ground_truth,
            positive_values_map=POSITIVE_VALUES_MAP,
            **runner_kwargs(seed=seed, alpha_pc=alpha_fallback),
        )

    # do()-style evaluation (reuse baseline brain)
    eval_seed = seed + 123
    df_base_eval, _ = generate_student_data(n_students=n_eval, seed=eval_seed, do=None)
    base_neuron, base_asm = _features_for_df(
        df_base_eval,
        brain=results['brain'],
        encoding_seed=seed,
        neurons_per_var=DEFAULTS.neurons_per_var,
        positive_prob=DEFAULTS.positive_prob,
        negative_prob=DEFAULTS.negative_prob,
    )

    base_raw_exam = _p_excellent(df_base_eval, 'FinalExamScore')
    base_raw_grade = _p_excellent(df_base_eval, 'FinalGrade')
    base_neuron_exam = float(base_neuron['FinalExamScore'].mean())
    base_neuron_grade = float(base_neuron['FinalGrade'].mean())
    base_asm_exam = float(base_asm['FinalExamScore'].mean())
    base_asm_grade = float(base_asm['FinalGrade'].mean())

    do_rows: list[dict] = []
    for label, do_dict in INTERVENTIONS:
        df_do, _ = generate_student_data(n_students=n_eval, seed=eval_seed, do=do_dict)
        do_neuron, do_asm = _features_for_df(
            df_do,
            brain=results['brain'],
            encoding_seed=seed,
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_prob=DEFAULTS.positive_prob,
            negative_prob=DEFAULTS.negative_prob,
        )

        raw_exam = _p_excellent(df_do, 'FinalExamScore')
        raw_grade = _p_excellent(df_do, 'FinalGrade')
        neuron_exam = float(do_neuron['FinalExamScore'].mean())
        neuron_grade = float(do_neuron['FinalGrade'].mean())
        asm_exam = float(do_asm['FinalExamScore'].mean())
        asm_grade = float(do_asm['FinalGrade'].mean())

        d_raw_exam = raw_exam - base_raw_exam
        d_raw_grade = raw_grade - base_raw_grade
        d_neuron_exam = neuron_exam - base_neuron_exam
        d_neuron_grade = neuron_grade - base_neuron_grade
        d_asm_exam = asm_exam - base_asm_exam
        d_asm_grade = asm_grade - base_asm_grade

        do_rows.append(
            {
                'seed': seed,
                'eval_seed': eval_seed,
                'intervention': label,
                'd_raw_exam': d_raw_exam,
                'd_raw_grade': d_raw_grade,
                'd_neuron_exam': d_neuron_exam,
                'd_neuron_grade': d_neuron_grade,
                'd_assembly_exam': d_asm_exam,
                'd_assembly_grade': d_asm_grade,
                'sign_match_neuron_exam': _sign(d_raw_exam) == _sign(d_neuron_exam),
                'sign_match_assembly_exam': _sign(d_raw_exam) == _sign(d_asm_exam),
                'sign_match_neuron_grade': _sign(d_raw_grade) == _sign(d_neuron_grade),
                'sign_match_assembly_grade': _sign(d_raw_grade) == _sign(d_asm_grade),
            }
        )

    # Seed-level summary
    summary = {
        'seed': seed,
        'n_samples': n_samples,
        'n_eval': n_eval,
        'alpha_used': used_alpha,
        'compression_ratio': float(results['compression']['ratio']),
        'neuron_f1': float(results['comparison']['neuron_vs_gt']['f1']),
        'assembly_f1': float(results['comparison']['assembly_vs_gt']['f1']),
        'preservation_score': float(results['comparison']['preservation_score']),
        'n_neuron_edges': int(len(results.get('neuron_edges', []))),
        'n_assembly_edges': int(len(results.get('assembly_edges', []))),
    }

    return summary, do_rows


def write_report(*, summary_df: pd.DataFrame, do_df: pd.DataFrame, out_md: str, summary_csv: str, do_csv: str) -> None:
    n = int(len(summary_df))

    neuron_f1 = summary_df['neuron_f1'].to_numpy(dtype=float)
    asm_f1 = summary_df['assembly_f1'].to_numpy(dtype=float)
    pres = summary_df['preservation_score'].to_numpy(dtype=float)

    # do() pooled correlations
    raw_exam = do_df['d_raw_exam'].to_numpy(dtype=float)
    raw_grade = do_df['d_raw_grade'].to_numpy(dtype=float)

    neuron_exam = do_df['d_neuron_exam'].to_numpy(dtype=float)
    neuron_grade = do_df['d_neuron_grade'].to_numpy(dtype=float)

    asm_exam = do_df['d_assembly_exam'].to_numpy(dtype=float)
    asm_grade = do_df['d_assembly_grade'].to_numpy(dtype=float)

    spearman_neuron_exam = _spearman(raw_exam, neuron_exam)
    spearman_asm_exam = _spearman(raw_exam, asm_exam)
    spearman_neuron_grade = _spearman(raw_grade, neuron_grade)
    spearman_asm_grade = _spearman(raw_grade, asm_grade)

    # do() sign agreement rates
    sign_neuron_exam = float(do_df['sign_match_neuron_exam'].mean())
    sign_asm_exam = float(do_df['sign_match_assembly_exam'].mean())
    sign_neuron_grade = float(do_df['sign_match_neuron_grade'].mean())
    sign_asm_grade = float(do_df['sign_match_assembly_grade'].mean())

    mu_nf1, lo_nf1, hi_nf1 = _ci95_mean(neuron_f1)
    mu_af1, lo_af1, hi_af1 = _ci95_mean(asm_f1)
    mu_ps, lo_ps, hi_ps = _ci95_mean(pres)

    defaults = asdict(DEFAULTS)

    lines: list[str] = []
    lines.append(f'# Student Success: multi-seed evaluation (n={n})')
    lines.append('')
    lines.append(f'- Seeds: {summary_df["seed"].min()} .. {summary_df["seed"].max()} (total {n})')
    lines.append(f'- Samples per seed: n_samples={int(summary_df["n_samples"].iloc[0])}, n_eval={int(summary_df["n_eval"].iloc[0])}')
    lines.append('')

    lines.append('## Defaults')
    lines.append('')
    lines.append('- ' + ', '.join([
        f"neurons_per_var={defaults['neurons_per_var']}",
        f"k={defaults['assembly_k']}",
        f"n_train={defaults['n_train']}",
        f"n_presentations={defaults['n_presentations']}",
        f"beta={defaults['beta']}",
        f"positive_prob={defaults['positive_prob']}",
        f"negative_prob={defaults['negative_prob']}",
    ]))
    lines.append('')

    lines.append('## 3-DAG metrics (mean with 95% CI)')
    lines.append('')
    lines.append(f'- Neuron DAG vs GT F1: {mu_nf1:.3f} [{lo_nf1:.3f}, {hi_nf1:.3f}]')
    lines.append(f'- Assembly DAG vs GT F1: {mu_af1:.3f} [{lo_af1:.3f}, {hi_af1:.3f}]')
    lines.append(f'- Preservation score: {mu_ps:.3f} [{lo_ps:.3f}, {hi_ps:.3f}]')
    lines.append('')

    lines.append('## do() effect preservation')
    lines.append('')
    lines.append('- Sign agreement (pooled across seeds x interventions):')
    lines.append(f'  - Exam: neuron={sign_neuron_exam:.3f}, assembly={sign_asm_exam:.3f}')
    lines.append(f'  - Grade: neuron={sign_neuron_grade:.3f}, assembly={sign_asm_grade:.3f}')
    lines.append('- Spearman correlation of effect deltas (pooled):')
    lines.append(f'  - Exam: raw vs neuron={spearman_neuron_exam:.3f}, raw vs assembly={spearman_asm_exam:.3f}')
    lines.append(f'  - Grade: raw vs neuron={spearman_neuron_grade:.3f}, raw vs assembly={spearman_asm_grade:.3f}')
    lines.append('')

    lines.append('## Artifacts')
    lines.append('')
    lines.append('- Summary CSV: `table_5_student_summary.csv`')
    lines.append('- do() deltas CSV: `table_5_student_dodeltas.csv`')
    lines.append('')

    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    # ------------------------------------------------------------------
    # MAGNITUDE / IMPORTANCE preservation (beyond direction)
    # ------------------------------------------------------------------
    # We look at whether larger raw effects correspond to larger feature effects.
    # This is done across all (seed, intervention) rows for stability.
    metrics = []
    for outcome_key, raw_col, n_col, a_col in [
        ('Exam', 'd_raw_exam', 'd_neuron_exam', 'd_assembly_exam'),
        ('Grade', 'd_raw_grade', 'd_neuron_grade', 'd_assembly_grade'),
    ]:
        raw = do_df[raw_col].to_numpy(dtype=float)
        neu = do_df[n_col].to_numpy(dtype=float)
        asm = do_df[a_col].to_numpy(dtype=float)

        # Correlation of signed deltas (linear association)
        r_neu = _corr(raw, neu)
        r_asm = _corr(raw, asm)

        # Rank correlation of absolute deltas (importance ordering)
        rho_neu = _spearman(np.abs(raw), np.abs(neu))
        rho_asm = _spearman(np.abs(raw), np.abs(asm))

        # Simple calibration: raw ≈ a·feature (no intercept)
        cal_neu = _ols_slope(neu, raw, fit_intercept=False)
        cal_asm = _ols_slope(asm, raw, fit_intercept=False)

        metrics.append({
            'Outcome': outcome_key,
            'PearsonR_raw_vs_neuron': r_neu,
            'PearsonR_raw_vs_assembly': r_asm,
            'SpearmanRho_abs_raw_vs_abs_neuron': rho_neu,
            'SpearmanRho_abs_raw_vs_abs_assembly': rho_asm,
            'CalibSlope_neuron_to_raw': cal_neu['a'],
            'CalibR2_neuron_to_raw': cal_neu['r2'],
            'CalibMAE_neuron_to_raw': cal_neu['mae'],
            'CalibSlope_assembly_to_raw': cal_asm['a'],
            'CalibR2_assembly_to_raw': cal_asm['r2'],
            'CalibMAE_assembly_to_raw': cal_asm['mae'],
        })

    mag_df = pd.DataFrame(metrics)

    lines.append('## Magnitude / importance preservation (beyond direction)')
    lines.append('')
    lines.append('These metrics ask whether *bigger raw do-effects correspond to bigger feature-space effects*, not just the same sign.')
    lines.append('')
    lines.append(_df_markdown_table(mag_df))
    lines.append('')
    lines.append('Notes: PearsonR uses signed deltas; SpearmanRho uses |delta| to test importance ordering; calibration fits raw ≈ a·feature.')
    lines.append('')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Multi-seed Student Success evaluation')
    parser.add_argument('--n-seeds', type=int, default=_ms.get('n_seeds', 50))
    parser.add_argument('--seed0', type=int, default=_ms.get('seed0', DEFAULTS.seed))
    parser.add_argument('--n-samples', type=int, default=_ms.get('n_samples', DEFAULTS.n_samples))
    parser.add_argument('--n-eval', type=int, default=_ms.get('n_eval', min(_ms.get('n_samples', DEFAULTS.n_samples), 2000)))
    parser.add_argument('--alpha-primary', type=float, default=_ms.get('alpha_primary', 0.1))
    parser.add_argument('--alpha-fallback', type=float, default=_ms.get('alpha_fallback', 0.2))
    parser.add_argument('--verbose', action='store_true', help='Print full per-seed logs')
    args = parser.parse_args()

    summaries: list[dict] = []
    do_rows: list[dict] = []

    seeds = [args.seed0 + i for i in range(args.n_seeds)]

    print(f'Running Student Success multi-seed evaluation: n_seeds={args.n_seeds}')
    print(f'  seed0={args.seed0} -> {seeds[-1]}')
    print(f'  n_samples={args.n_samples}, n_eval={args.n_eval}')
    print(f'  alpha_primary={args.alpha_primary}, alpha_fallback={args.alpha_fallback}')

    for idx, seed in enumerate(seeds, start=1):
        if (idx == 1) or (idx % 5 == 0) or (idx == len(seeds)):
            print(f'  [{idx:02d}/{len(seeds):02d}] seed={seed}')

        if args.verbose:
            summary, do_part = run_seed(
                seed=seed,
                n_samples=args.n_samples,
                n_eval=args.n_eval,
                alpha_primary=args.alpha_primary,
                alpha_fallback=args.alpha_fallback,
            )
        else:
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                summary, do_part = run_seed(
                    seed=seed,
                    n_samples=args.n_samples,
                    n_eval=args.n_eval,
                    alpha_primary=args.alpha_primary,
                    alpha_fallback=args.alpha_fallback,
                )
        summaries.append(summary)
        do_rows.extend(do_part)

    summary_df = pd.DataFrame(summaries).sort_values('seed')
    do_df = pd.DataFrame(do_rows).sort_values(['seed', 'intervention'])

    summary_csv = get_output_filepath('table_5_student_summary.csv')
    do_csv = get_output_filepath('table_5_student_dodeltas.csv')
    summary_df.to_csv(summary_csv, index=False)
    do_df.to_csv(do_csv, index=False)

    out_md = get_output_filepath('table_5_student_report.md')
    write_report(summary_df=summary_df, do_df=do_df, out_md=out_md, summary_csv=summary_csv, do_csv=do_csv)

    print('Done.')
    print(f'  Wrote: {summary_csv}')
    print(f'  Wrote: {do_csv}')
    print(f'  Wrote: {out_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
























