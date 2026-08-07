"""
Credit Default Case Study - 3-DAG Causal Validation

Validates that neural assemblies preserve causal information by comparing:
1. Ground Truth DAG (from finance literature)
2. Neuron DAG (causal discovery on neuron activations)
3. Assembly DAG (causal discovery on assembly activations)

Ground Truth Causal Structure (5 variables):
  Income -> CreditUtilization
  PaymentHistory -> Default
  CreditUtilization -> Default
  DebtToIncome -> Default
"""

import json
import datetime

import numpy as np
import pandas as pd
import os
from experiments.experiment_defaults import DEFAULTS, runner_kwargs, get_output_filepath
from src.runner import run_causal_dag_validation
from src.visualization.dag_plotting import visualize_three_dags
from src.validation.interventional.pearl_do_calculus import (
    compute_neuron_and_assembly_mean_features,
    probability_of_values,
    sign,
)


def generate_credit_data(n_customers=DEFAULTS.n_samples, seed=DEFAULTS.seed, do=None):
    """
    Generate credit default data with STRONG realistic causal effects.

    Based on finance literature showing payment history and credit utilization
    are the strongest predictors of default.

    Ground Truth:
      Income -> CreditUtilization
      PaymentHistory -> Default (VERY STRONG)
      CreditUtilization -> Default (STRONG)
      DebtToIncome -> Default
    """
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print("GENERATING CREDIT DEFAULT DATA")
    print("=" * 70)
    print(f"  Cohort size: {n_customers} customers")
    print(
        f"  Variables: 5 (Income, PaymentHistory, CreditUtilization, DebtToIncome, Default)"
    )
    if do:
        print(f"  Intervention: do({do})")

    data = []

    for _ in range(n_customers):
        # Baseline characteristics (exogenous)
        income = np.random.choice(["Low", "High"], p=[0.40, 0.60])
        payment_history = np.random.choice(["Good", "LatePay"], p=[0.70, 0.30])
        debt_to_income = np.random.choice(["Low", "High"], p=[0.55, 0.45])

        if do:
            if "Income" in do:
                income = do["Income"]
            if "PaymentHistory" in do:
                payment_history = do["PaymentHistory"]
            if "DebtToIncome" in do:
                debt_to_income = do["DebtToIncome"]

        # Credit Utilization (caused by Income)
        # Low income -> harder to pay down balances -> high utilization
        p_high_util = 0.20  # Baseline
        if income == "Low":
            p_high_util += 0.40  # Strong effect

        credit_utilization = np.random.choice(
            ["Low", "High"], p=[1 - p_high_util, p_high_util]
        )

        if do and "CreditUtilization" in do:
            credit_utilization = do["CreditUtilization"]

        # Default (caused by PaymentHistory, CreditUtilization, DebtToIncome)
        # Literature shows payment history is THE strongest predictor
        p_default = 0.05  # Low baseline (5% default rate)

        if payment_history == "LatePay":
            p_default += 0.50  # VERY STRONG effect (late payments -> default)

        if credit_utilization == "High":
            p_default += 0.25  # STRONG effect (maxed out cards -> default)

        if debt_to_income == "High":
            p_default += 0.15  # Medium effect (overleveraged -> default)

        default = np.random.choice(["NoDef", "Default"], p=[1 - p_default, p_default])

        if do and "Default" in do:
            default = do["Default"]

        data.append(
            {
                "Income": income,
                "PaymentHistory": payment_history,
                "CreditUtilization": credit_utilization,
                "DebtToIncome": debt_to_income,
                "Default": default,
            }
        )

    df = pd.DataFrame(data)

    # Ground truth DAG
    ground_truth_edges = [
        ("Income", "CreditUtilization"),
        ("PaymentHistory", "Default"),
        ("CreditUtilization", "Default"),
        ("DebtToIncome", "Default"),
    ]

    print("\nGround Truth Causal Structure:")
    print("  (Based on FICO scoring and finance literature)")
    for source, target in ground_truth_edges:
        print(f"  {source} -> {target}")

    print("\nOutcome Distribution:")
    print(df["Default"].value_counts())
    print(f'  Default Rate: {(df["Default"] == "Default").mean() * 100:.1f}%')
    print(
        f'  High Credit Utilization: {(df["CreditUtilization"] == "High").mean() * 100:.1f}%'
    )

    return df, ground_truth_edges


if __name__ == "__main__":
    # Prepare output file
    csv_path = get_output_filepath("table_1_credit_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Generate data
    df, ground_truth_edges = generate_credit_data(
        n_customers=DEFAULTS.n_samples, seed=DEFAULTS.seed
    )

    # Define positive values for correct encoding (risk factors)
    positive_values_map = {
        "Income": {"Low"},
        "PaymentHistory": {"LatePay"},
        "CreditUtilization": {"High"},
        "DebtToIncome": {"High"},
        "Default": {"Default"},
    }

    # Run validation
    results = run_causal_dag_validation(
        df=df,
        var_names=list(df.columns),
        ground_truth_edges=ground_truth_edges,
        positive_values_map=positive_values_map,
        **runner_kwargs(),
    )

    # ---------------------------------------------------------------------
    # do-operator intervention evaluation (effect preservation)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERVENTION EVALUATION (do operator)")
    print("=" * 70)
    print(
        "  Goal: compare interventional outcome differences in raw space versus feature space"
    )
    print("  Note: reuse the trained Brain from the baseline run (no retraining)")

    n_eval = min(DEFAULTS.n_samples, 2000)
    eval_seed = DEFAULTS.seed + 123

    df_base_eval, _ = generate_credit_data(n_customers=n_eval, seed=eval_seed, do=None)
    var_names = list(df_base_eval.columns)

    base_neuron_means, base_assembly_means, assembly_flip_map = (
        compute_neuron_and_assembly_mean_features(
            df_base_eval,
            brain=results["brain"],
            var_names=var_names,
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=DEFAULTS.positive_prob,
            negative_prob=DEFAULTS.negative_prob,
            seed=DEFAULTS.seed,
            return_assembly_flip_map=True,
        )
    )

    baseline_raw = probability_of_values(df_base_eval, "Default", {"Default"})
    baseline_neuron = base_neuron_means["Default"]
    baseline_assembly = base_assembly_means["Default"]

    print(f"  Eval cohort size: {n_eval} (seed={eval_seed})")
    print(
        f"  Baseline: P(Default=Default)={baseline_raw:.3f}, neuron mean={baseline_neuron:.3f}, assembly mean={baseline_assembly:.3f}"
    )

    interventions = [
        ("do(PaymentHistory=LatePay)", {"PaymentHistory": "LatePay"}),
        ("do(PaymentHistory=Good)", {"PaymentHistory": "Good"}),
        ("do(CreditUtilization=High)", {"CreditUtilization": "High"}),
        ("do(DebtToIncome=High)", {"DebtToIncome": "High"}),
        ("do(Income=Low)", {"Income": "Low"}),
    ]

    print("\nIntervention effect differences (do intervention minus baseline)")
    print(
        "Intervention                          | raw change | neuron change | assembly change | sign match (Neuron/Assembly)"
    )
    print(
        "  Interpretation: N:Y/A:Y => same direction as raw change (intervention effect preserved directionally); N:N or A:N => mismatch. Magnitudes can differ due to different scales."
    )
    print("-" * 110)

    total = 0
    ok_neuron = 0
    ok_assembly = 0

    sign_match_summary = None
    for label, do_dict in interventions:
        df_do_eval, _ = generate_credit_data(
            n_customers=n_eval, seed=eval_seed, do=do_dict
        )
        do_neuron_means, do_assembly_means = compute_neuron_and_assembly_mean_features(
            df_do_eval,
            brain=results["brain"],
            var_names=var_names,
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=DEFAULTS.positive_prob,
            negative_prob=DEFAULTS.negative_prob,
            seed=DEFAULTS.seed,
            assembly_flip_map=assembly_flip_map,
        )

        raw = probability_of_values(df_do_eval, "Default", {"Default"})
        neuron = do_neuron_means["Default"]
        assembly = do_assembly_means["Default"]

        d_raw = raw - baseline_raw
        d_neuron = neuron - baseline_neuron
        d_assembly = assembly - baseline_assembly

        match_neuron = "Y" if sign(d_raw) == sign(d_neuron) else "N"
        match_assembly = "Y" if sign(d_raw) == sign(d_assembly) else "N"

        total += 1
        if match_neuron == "Y":
            ok_neuron += 1
        if match_assembly == "Y":
            ok_assembly += 1

        print(
            f"{label:36s} | {d_raw:+.3f}      | {d_neuron:+.3f}        | {d_assembly:+.3f}        | N:{match_neuron}/A:{match_assembly}"
        )

    if total > 0:
        sign_match_summary = (
            f"Neuron={ok_neuron}/{total} ({ok_neuron/total:.1%}), "
            f"Assembly={ok_assembly}/{total} ({ok_assembly/total:.1%})"
        )
        print()
        print(f"  Sign-match rate (directional): {sign_match_summary}")
    # Visualize comparison
    visualize_three_dags(
        ground_truth_edges=ground_truth_edges,
        neuron_edges=results["neuron_edges"],
        assembly_edges=results["assembly_edges"],
        var_names=list(df.columns),
        save_path=os.path.join(os.path.dirname(csv_path), "credit_3dag_comparison.png"),
    )

    # Save detailed results
    with open(
        os.path.join(os.path.dirname(csv_path), "credit_causal_results.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("=" * 70 + "\n")
        f.write("CREDIT DEFAULT - 3-DAG CAUSAL VALIDATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("GROUND TRUTH EDGES:\n")
        for source, target in results["ground_truth_edges"]:
            f.write(f"  {source} -> {target}\n")

        f.write(f'\nNEURON DAG EDGES ({len(results["neuron_edges"])} discovered):\n')
        for source, target in results["neuron_edges"]:
            f.write(f"  {source} -> {target}\n")

        f.write(
            f'\nASSEMBLY DAG EDGES ({len(results["assembly_edges"])} discovered):\n'
        )
        for source, target in results["assembly_edges"]:
            f.write(f"  {source} -> {target}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("VALIDATION METRICS:\n")
        f.write("=" * 70 + "\n")

        f.write("\nNeuron DAG vs Ground Truth:\n")
        ncomp = results["comparison"]["neuron_vs_gt"]
        f.write(f'  Precision: {ncomp["precision"]:.3f}\n')
        f.write(f'  Recall: {ncomp["recall"]:.3f}\n')
        f.write(f'  F1 Score: {ncomp["f1"]:.3f}\n')

        f.write("\nAssembly DAG vs Ground Truth:\n")
        acomp = results["comparison"]["assembly_vs_gt"]
        f.write(f'  Precision: {acomp["precision"]:.3f}\n')
        f.write(f'  Recall: {acomp["recall"]:.3f}\n')
        f.write(f'  F1 Score: {acomp["f1"]:.3f}\n')

        f.write("\nAssembly vs Neuron Agreement:\n")
        ancomp = results["comparison"]["assembly_vs_neuron"]
        f.write(f'  Similarity: {ancomp["precision"]:.3f}\n')

        f.write(
            f'\nOverall Preservation Score: {results["comparison"]["preservation_score"]:.3f}\n'
        )
        f.write(f'Compression Ratio: {results["compression"]["ratio"]:.1f}x\n')

        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERVENTION SIGN-MATCH (do operator):\n")
        f.write("=" * 70 + "\n")
        if sign_match_summary:
            f.write(f"Sign-match rate (directional): {sign_match_summary}\n")
        else:
            f.write("Sign-match rate (directional): (not computed)\n")

        # Data statistics
        f.write("\n" + "=" * 70 + "\n")
        f.write("DATA STATISTICS:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sample Size: {DEFAULTS.n_samples} customers\n")
        f.write(f'Default Rate: {(df["Default"] == "Default").mean() * 100:.1f}%\n')
        f.write(
            f'High Credit Utilization: {(df["CreditUtilization"] == "High").mean() * 100:.1f}%\n'
        )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(
        f"  Results: {os.path.join(os.path.dirname(csv_path), 'credit_causal_results.txt')}"
    )
    print(
        f"  Visualization: {os.path.join(os.path.dirname(csv_path), 'credit_3dag_comparison.png')}"
    )
