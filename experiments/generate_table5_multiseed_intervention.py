"""Multi-seed interventional sign-match evaluation across all five paper datasets.

Usage:
  python -m experiments.generate_table5_multiseed_intervention
  python -m experiments.generate_table5_multiseed_intervention --n-seeds 10 --seed-start 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_defaults import DEFAULTS, runner_kwargs, get_output_filepath
from src.runner import run_causal_dag_validation
from src.encoding.bernoulli import encode_bernoulli_dataframe
from src.representation.assembly_feature_extraction import extract_assembly_features

from experiments.alzheimers.validate_alzheimers import generate_alzheimers_data
from experiments.stroke_risk.validate_stroke import generate_stroke_data
from experiments.credit_default.validate_credit import generate_credit_data
from experiments.student_success.validate_student_success import generate_student_data
from experiments.vaccine_efficacy.validate_vaccine import generate_vaccine_data
from experiments.generate_table3_multiseed_run_artifacts import build_datasets


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _positive_value(var: str, positive_values_map: dict) -> str:
    return next(iter(positive_values_map[var]))


def _features_for_df(df, *, brain, var_names, positive_values_map, neurons_per_var,
                     positive_prob, negative_prob, encoding_seed):
    neural = encode_bernoulli_dataframe(
        df, var_names, neurons_per_var=neurons_per_var,
        positive_values_map=positive_values_map,
        positive_prob=positive_prob, negative_prob=negative_prob, seed=encoding_seed)
    feats = {}
    for i, var in enumerate(var_names):
        feats[var] = neural[:, i * neurons_per_var:(i + 1) * neurons_per_var].mean(axis=1)
    neuron_df = pd.DataFrame(feats)
    src_map = {v: v for v in var_names}
    asm_raw = extract_assembly_features(neural, brain, var_names, neurons_per_var,
                                        source_var_by_target_area_name=src_map)
    assembly_df = pd.DataFrame({k: v for k, v in asm_raw.items() if "_x_" not in k})
    return neuron_df, assembly_df


# dataset registry: uniform generate(n_samples, seed, do) interface
_GENERATORS = {
    "Alzheimers": lambda n, s, do: generate_alzheimers_data(n_patients=n, seed=s, do=do),
    "Stroke":     lambda n, s, do: generate_stroke_data(n_patients=n, seed=s, do=do),
    "Credit":     lambda n, s, do: generate_credit_data(n_customers=n, seed=s, do=do),
    "Student":    lambda n, s, do: generate_student_data(n_students=n, seed=s, do=do),
    "Vaccine":    lambda n, s, do: generate_vaccine_data(n_patients=n, seed=s, do=do),
}


def _run_seed_for_dataset(*, dataset_name: str, seed: int, n_samples: int, n_eval: int,
                          alpha: float, ds: dict) -> list[dict]:
    gen = _GENERATORS[dataset_name]
    var_names = ds["var_names"]
    pvm = ds["positive_values_map"]

    df, gt_edges = gen(n_samples, seed, None)

    result = run_causal_dag_validation(
        df=df, var_names=var_names, ground_truth_edges=gt_edges,
        positive_values_map=pvm,
        **runner_kwargs(seed=seed, alpha_pc=alpha))

    # outcome = leaf node (no outgoing edges)
    sources = {e[0] for e in gt_edges}
    outcome = next(v for v in reversed(var_names) if v not in sources)

    # interventions: set each non-outcome node to its positive value
    intervene_vars = [v for v in var_names if v != outcome]
    eval_seed = seed + 123
    df_base, _ = gen(n_eval, eval_seed, None)
    base_n, base_a = _features_for_df(df_base, brain=result["brain"],
        var_names=var_names, positive_values_map=pvm,
        neurons_per_var=DEFAULTS.neurons_per_var,
        positive_prob=DEFAULTS.positive_prob, negative_prob=DEFAULTS.negative_prob,
        encoding_seed=seed)
    base_raw  = float((df_base[outcome] == _positive_value(outcome, pvm)).mean())
    base_neu  = float(base_n[outcome].mean())
    base_asm  = float(base_a[outcome].mean())

    rows = []
    for var in intervene_vars:
        do_val  = _positive_value(var, pvm)
        df_do, _ = gen(n_eval, eval_seed, {var: do_val})
        do_n, do_a = _features_for_df(df_do, brain=result["brain"],
            var_names=var_names, positive_values_map=pvm,
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_prob=DEFAULTS.positive_prob, negative_prob=DEFAULTS.negative_prob,
            encoding_seed=seed)
        d_raw = float((df_do[outcome] == _positive_value(outcome, pvm)).mean()) - base_raw
        d_neu = float(do_n[outcome].mean()) - base_neu
        d_asm = float(do_a[outcome].mean()) - base_asm
        rows.append({
            "dataset":              dataset_name,
            "seed":                 seed,
            "intervention":         f"do({var}={do_val})",
            "outcome_var":          outcome,
            "d_raw":                d_raw,
            "d_neuron":             d_neu,
            "d_assembly":           d_asm,
            "sign_match_neuron":    _sign(d_raw) == _sign(d_neu),
            "sign_match_assembly":  _sign(d_raw) == _sign(d_asm),
            "neuron_f1":            float(result["comparison"]["neuron_vs_gt"]["f1"]),
            "assembly_f1":          float(result["comparison"]["assembly_vs_gt"]["f1"]),
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds",   type=int, default=10)  # paper used 50; increase with --n-seeds 50
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=DEFAULTS.n_samples)
    p.add_argument("--n-eval",    type=int, default=DEFAULTS.n_samples)
    p.add_argument("--alpha",     type=float, default=0.1)
    p.add_argument("--datasets",  type=str, default="Alzheimers,Stroke,Credit,Student,Vaccine")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    selected = [d.strip() for d in args.datasets.split(",")]
    ds_all = build_datasets(n_samples=args.n_samples)

    all_rows: list[dict] = []
    total = len(selected) * len(seeds)
    idx = 0
    for dname in selected:
        for seed in seeds:
            idx += 1
            print(f"[{idx}/{total}] {dname} seed={seed}")
            rows = _run_seed_for_dataset(
                dataset_name=dname, seed=seed,
                n_samples=args.n_samples, n_eval=args.n_eval,
                alpha=args.alpha, ds=ds_all[dname])
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = get_output_filepath("all_datasets_multiseed_intervention.csv")
    df.to_csv(out_path, index=False)

    # sign-match summary per dataset
    summary = (df.groupby("dataset")
               .agg(seeds=("seed", "nunique"),
                    interventions=("intervention", "nunique"),
                    n_tests=("sign_match_assembly", "count"),
                    sign_match_neuron_rate=("sign_match_neuron", "mean"),
                    sign_match_assembly_rate=("sign_match_assembly", "mean"),
                    mean_neuron_f1=("neuron_f1", "mean"),
                    mean_assembly_f1=("assembly_f1", "mean"))
               .reset_index())
    summary_path = get_output_filepath("all_datasets_multiseed_intervention_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nDetailed results: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
