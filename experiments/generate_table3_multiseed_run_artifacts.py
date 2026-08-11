"""Encoding Sensitivity Sweep.

Full method name: encoding sensitivity sweep for neural assembly causal
discovery under encoding uncertainty.

How it works: iterate over SCM datasets, random seeds, and encoding settings;
call src.runner.run_causal_dag_validation with PC or GES; collect neuron and
assembly DAG F1 scores; write raw, summary, and overall CSV outputs.

Usage examples:
  python -m experiments.generate_table3_multiseed_run_artifacts
  python -m experiments.generate_table3_multiseed_run_artifacts --datasets Credit,Student,Vaccine --method ges --n-samples 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.experiment_defaults import DEFAULTS, get_output_filepath, _cfg
from src.runner import run_causal_dag_validation

_ss = _cfg.get("sensitivity_sweep", {})

from experiments.alzheimers.validate_alzheimers import generate_alzheimers_data
from experiments.stroke_risk.validate_stroke import generate_stroke_data
from experiments.credit_default.validate_credit import generate_credit_data
from experiments.student_success.validate_student_success import generate_student_data
from experiments.vaccine_efficacy.validate_vaccine import generate_vaccine_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run encoding sensitivity sweep")
    parser.add_argument(
        "--datasets",
        default=_ss.get("datasets", "Alzheimers,Stroke,Credit,Student,Vaccine"),
    )
    parser.add_argument("--seeds", default=_ss.get("seeds", "42,73,101,131,151,181,211,241,271,301"))
    parser.add_argument("--n-samples", type=int, default=_ss.get("n_samples", 800))
    parser.add_argument("--n-train", type=int, default=_ss.get("n_train", 120))
    parser.add_argument(
        "--method",
        choices=["pc", "ges"],
        default=_ss.get("method", "pc"),
        help="Causal discovery method",
    )
    parser.add_argument(
        "--n-presentations", type=int, default=_ss.get("n_presentations", 3)
    )
    parser.add_argument("--alpha-pc", type=float, default=_ss.get("alpha_pc", 0.05))
    return parser.parse_args()


def _f1(comp: dict, key: str) -> float:
    sub = (comp or {}).get(key) or {}
    return float(sub.get("f1", float("nan")))


def build_datasets(n_samples: int):
    return {
        "Alzheimers": {
            "gen": lambda seed: generate_alzheimers_data(
                n_patients=n_samples, seed=seed
            ),
            "var_names": [
                "APOE4",
                "PhysicalActivity",
                "Diet",
                "Cholesterol",
                "Inflammation",
                "SleepQuality",
                "Education",
                "Amyloid",
                "Tau",
                "CognitiveDecline",
            ],
            "positive_values_map": {
                "APOE4": {"Carrier"},
                "PhysicalActivity": {"Sedentary"},
                "Diet": {"Poor"},
                "Cholesterol": {"High"},
                "Inflammation": {"High"},
                "SleepQuality": {"Poor"},
                "Education": {"Low"},
                "Amyloid": {"High"},
                "Tau": {"High"},
                "CognitiveDecline": {"Decline"},
            },
        },
        "Stroke": {
            "gen": lambda seed: generate_stroke_data(n_patients=n_samples, seed=seed),
            "var_names": [
                "Hypertension",
                "Age",
                "Smoking",
                "Atherosclerosis",
                "BloodClotting",
                "Stroke",
            ],
            "positive_values_map": {
                "Hypertension": {"High"},
                "Age": {"Elderly"},
                "Smoking": {"Smoker"},
                "Atherosclerosis": {"Present"},
                "BloodClotting": {"Hypercoagulable"},
                "Stroke": {"Stroke"},
            },
        },
        "Credit": {
            "gen": lambda seed: generate_credit_data(n_customers=n_samples, seed=seed),
            "var_names": [
                "Income",
                "PaymentHistory",
                "CreditUtilization",
                "DebtToIncome",
                "Default",
            ],
            "positive_values_map": {
                "Income": {"Low"},
                "PaymentHistory": {"LatePay"},
                "CreditUtilization": {"High"},
                "DebtToIncome": {"High"},
                "Default": {"Default"},
            },
        },
        "Student": {
            "gen": lambda seed: generate_student_data(n_students=n_samples, seed=seed),
            "var_names": [
                "PriorGPA",
                "Attendance",
                "StudyHours",
                "FinalExamScore",
                "FinalGrade",
            ],
            "positive_values_map": {
                "PriorGPA": {"High"},
                "Attendance": {"Good"},
                "StudyHours": {"High"},
                "FinalExamScore": {"Excellent"},
                "FinalGrade": {"Excellent"},
            },
        },
        "Vaccine": {
            "gen": lambda seed: generate_vaccine_data(n_patients=n_samples, seed=seed),
            "var_names": [
                "Vaccination",
                "ImmuneHealth",
                "AntibodyResponse",
                "Immunity",
            ],
            "positive_values_map": {
                "Vaccination": {"Vaccinated"},
                "ImmuneHealth": {"Strong"},
                "AntibodyResponse": {"High"},
                "Immunity": {"Immune"},
            },
        },
    }


def build_sweep_configs(base_kwargs: dict) -> list[dict]:
    configs: list[dict] = []

    for pos in [0.20, 0.30, 0.40]:
        configs.append(
            {
                "name": f"Bernoulli_pos{pos:.2f}_neg0.10",
                "kwargs": {
                    **base_kwargs,
                    "deterministic_k_encoding": False,
                    "positive_prob": pos,
                    "negative_prob": 0.10,
                },
            }
        )
    for neg in [0.05, 0.10, 0.15]:
        configs.append(
            {
                "name": f"Bernoulli_pos0.30_neg{neg:.2f}",
                "kwargs": {
                    **base_kwargs,
                    "deterministic_k_encoding": False,
                    "positive_prob": 0.30,
                    "negative_prob": neg,
                },
            }
        )

    for step in [5, 10, 15]:
        configs.append(
            {
                "name": f"Deterministic_kstep{step}",
                "kwargs": {
                    **base_kwargs,
                    "deterministic_k_encoding": True,
                    "deterministic_k_step": step,
                    "deterministic_k_readout_mode": "pool_mean",
                    "stimulus_k": DEFAULTS.assembly_k,
                },
            }
        )

    dedup: dict[str, dict] = {}
    for c in configs:
        dedup[c["name"]] = c
    return list(dedup.values())


def main() -> None:
    args = parse_args()
    selected_datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    datasets = build_datasets(args.n_samples)
    missing = [d for d in selected_datasets if d not in datasets]
    if missing:
        raise ValueError(f"Unknown datasets: {missing}")

    base_kwargs = dict(
        neurons_per_var=DEFAULTS.neurons_per_var,
        assembly_k=DEFAULTS.assembly_k,
        n_train=args.n_train,
        n_presentations=args.n_presentations,
        beta=DEFAULTS.beta,
        alpha_pc=args.alpha_pc,
        strict_neuron_causal_discovery=False,
        strict_assembly_causal_discovery=False,
    )

    sweep_configs = build_sweep_configs(base_kwargs)

    rows: list[dict] = []
    total = len(selected_datasets) * len(seeds) * len(sweep_configs)
    idx = 0

    print(
        f"Running sensitivity sweep using method={args.method}: datasets={selected_datasets}, seeds={seeds}, configs={len(sweep_configs)}, total runs={total}"
    )

    for ds_name in selected_datasets:
        ds = datasets[ds_name]
        for seed in seeds:
            df, gt_edges = ds["gen"](seed)
            for cfg in sweep_configs:
                idx += 1
                print(f"[{idx}/{total}] {ds_name} | seed={seed} | {cfg['name']}")

                kwargs = {**cfg["kwargs"], "seed": seed}
                results = run_causal_dag_validation(
                    method=args.method,
                    df=df,
                    var_names=ds["var_names"],
                    ground_truth_edges=gt_edges,
                    positive_values_map=ds["positive_values_map"],
                    **kwargs,
                )

                comp = results.get("comparison", {})
                nf = _f1(comp, "neuron_vs_gt")
                af = _f1(comp, "assembly_vs_gt")

                rows.append(
                    {
                        "Dataset": ds_name,
                        "Method": args.method,
                        "Seed": seed,
                        "Config": cfg["name"],
                        "Neuron_F1": nf,
                        "Assembly_F1": af,
                        "Gap_AssemblyMinusNeuron": af - nf,
                        "GT_Edges": len(gt_edges),
                        "Neuron_Edges": len(results.get("neuron_edges") or []),
                        "Assembly_Edges": len(results.get("assembly_edges") or []),
                    }
                )

    if args.method == "pc":
        raw_name = "table_3_pc_raw.csv"
        summary_name = "table_3_pc_summary.csv"
        overall_name = "table_3_pc_overall.csv"
    else:
        raw_name = "table_3_ges_raw.csv"
        summary_name = "table_3_ges_summary.csv"
        overall_name = "table_3_ges_overall.csv"

    df_all = pd.DataFrame(rows)
    all_path = Path(get_output_filepath(raw_name))
    df_all.to_csv(all_path, index=False)

    summary = (
        df_all.groupby(["Dataset", "Config"], as_index=False)
        .agg(
            Neuron_F1_mean=("Neuron_F1", "mean"),
            Neuron_F1_std=("Neuron_F1", "std"),
            Assembly_F1_mean=("Assembly_F1", "mean"),
            Assembly_F1_std=("Assembly_F1", "std"),
            Gap_mean=("Gap_AssemblyMinusNeuron", "mean"),
            Gap_std=("Gap_AssemblyMinusNeuron", "std"),
            Runs=("Neuron_F1", "count"),
        )
        .sort_values(["Dataset", "Assembly_F1_mean"], ascending=[True, False])
    )

    summary_path = Path(get_output_filepath(summary_name))
    summary.to_csv(summary_path, index=False)

    overall = (
        df_all.groupby("Config", as_index=False)
        .agg(
            Neuron_F1_mean=("Neuron_F1", "mean"),
            Assembly_F1_mean=("Assembly_F1", "mean"),
            Gap_mean=("Gap_AssemblyMinusNeuron", "mean"),
        )
        .sort_values("Assembly_F1_mean", ascending=False)
    )

    overall_path = Path(get_output_filepath(overall_name))
    overall.to_csv(overall_path, index=False)

    print("\n" + "=" * 80)
    print("OVERALL CONFIG RANKING (higher Assembly_F1_mean is better)")
    print("=" * 80)
    print(overall.to_string(index=False))

    print(f"\nSaved raw: {all_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved overall: {overall_path}")


if __name__ == "__main__":
    main()




