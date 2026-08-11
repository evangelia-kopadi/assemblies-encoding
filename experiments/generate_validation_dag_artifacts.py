"""Generate validation diagnostics for all methods and encodings.

Produces `*_causal_results.txt` and `*_3dag_comparison.png` artifacts for every
dataset under:
- methods: PC, GES
- encodings: deterministic-k(step=10), Bernoulli(0.30/0.10)

Usage:
  python -m experiments.generate_validation_dag_artifacts
  python -m experiments.generate_validation_dag_artifacts --methods pc --datasets Credit,Student
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.experiment_defaults import DEFAULTS, get_output_filepath, runner_kwargs
from experiments.generate_table3_multiseed_run_artifacts import build_datasets
from src.runner import run_causal_dag_validation
from src.visualization.dag_plotting import visualize_three_dags


ENCODINGS = [
    (
        "detk",
        "Deterministic-k (step 10)",
        {
            "deterministic_k_encoding": True,
            "deterministic_k_step": 10,
            "deterministic_k_readout_mode": "pool_mean",
            "stimulus_k": DEFAULTS.assembly_k,
        },
    ),
    (
        "bernoulli",
        "Bernoulli (0.30/0.10)",
        {
            "deterministic_k_encoding": False,
            "positive_prob": 0.30,
            "negative_prob": 0.10,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate causal_results and 3DAG diagnostics for all method/encoding combinations"
    )
    parser.add_argument(
        "--datasets",
        default="Alzheimers,Stroke,Credit,Student,Vaccine",
        help="Comma-separated datasets",
    )
    parser.add_argument(
        "--methods",
        default="pc,ges",
        help="Comma-separated methods (allowed: pc,ges)",
    )
    parser.add_argument("--n-samples", type=int, default=DEFAULTS.n_samples)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    return parser.parse_args()


def _normalize_methods(raw: str) -> list[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in methods if m not in {"pc", "ges"}]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}")
    seen: set[str] = set()
    out: list[str] = []
    for m in methods:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _write_text_report(path: Path, *, dataset: str, method: str, encoding_label: str, result: dict) -> None:
    comp = result.get("comparison", {})
    ng = comp.get("neuron_vs_gt", {})
    ag = comp.get("assembly_vs_gt", {})
    na = comp.get("neuron_vs_assembly", {})

    lines = [
        f"Dataset: {dataset}",
        f"Method: {method.upper()}",
        f"Encoding: {encoding_label}",
        "",
        "NEURON vs GT",
        f"Precision: {float(ng.get('precision', 0.0)):.3f}",
        f"Recall: {float(ng.get('recall', 0.0)):.3f}",
        f"F1: {float(ng.get('f1', 0.0)):.3f}",
        "",
        "ASSEMBLY vs GT",
        f"Precision: {float(ag.get('precision', 0.0)):.3f}",
        f"Recall: {float(ag.get('recall', 0.0)):.3f}",
        f"F1: {float(ag.get('f1', 0.0)):.3f}",
        "",
        "NEURON vs ASSEMBLY",
        f"Agreement precision: {float(na.get('precision', 0.0)):.3f}",
        f"Agreement recall: {float(na.get('recall', 0.0)):.3f}",
        f"Agreement F1: {float(na.get('f1', 0.0)):.3f}",
        "",
        f"GT edges: {len(result.get('ground_truth_edges') or [])}",
        f"Neuron edges: {len(result.get('neuron_edges') or [])}",
        f"Assembly edges: {len(result.get('assembly_edges') or [])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = _normalize_methods(args.methods)

    datasets = build_datasets(args.n_samples)
    missing = [d for d in selected_datasets if d not in datasets]
    if missing:
        raise ValueError(f"Unknown datasets: {missing}")

    total = len(selected_datasets) * len(methods) * len(ENCODINGS)
    idx = 0
    print(
        f"Generating validation matrix: datasets={selected_datasets}, methods={methods}, "
        f"encodings={len(ENCODINGS)}, total runs={total}"
    )

    for dataset_name in selected_datasets:
        ds = datasets[dataset_name]
        df, gt_edges = ds["gen"](args.seed)
        ds_slug = _slug(dataset_name)

        for method in methods:
            for enc_slug, enc_label, enc_overrides in ENCODINGS:
                idx += 1
                print(f"[{idx}/{total}] {dataset_name} | {method.upper()} | {enc_label}")

                result = run_causal_dag_validation(
                    method=method,
                    df=df,
                    var_names=ds["var_names"],
                    ground_truth_edges=gt_edges,
                    positive_values_map=ds["positive_values_map"],
                    strict_neuron_causal_discovery=False,
                    strict_assembly_causal_discovery=False,
                    **runner_kwargs(seed=args.seed, **enc_overrides),
                )

                base_name = f"{ds_slug}_{method}_{enc_slug}"
                txt_path = Path(get_output_filepath(f"{base_name}_causal_results.txt"))
                png_path = Path(get_output_filepath(f"{base_name}_3dag_comparison.png"))

                _write_text_report(
                    txt_path,
                    dataset=dataset_name,
                    method=method,
                    encoding_label=enc_label,
                    result=result,
                )

                visualize_three_dags(
                    ground_truth_edges=result.get("ground_truth_edges") or [],
                    neuron_edges=result.get("neuron_edges") or [],
                    assembly_edges=result.get("assembly_edges") or [],
                    var_names=ds["var_names"],
                    save_path=str(png_path),
                )

                print(f"  Saved: {txt_path.name}, {png_path.name}")


if __name__ == "__main__":
    main()



