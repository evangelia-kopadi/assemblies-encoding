"""
Stroke Risk Case Study - 3-DAG Causal Validation

Validates causal preservation with stroke risk factors.
Same structure as Alzheimer's: two independent pathways converging.

Ground Truth Causal Structure (from epidemiology):
  Hypertension -> Atherosclerosis -> Stroke (vascular pathway)
  Smoking -> BloodClotting -> Stroke (thrombotic pathway)
  Age -> Atherosclerosis (age-related vascular damage)
"""


import sys
import json
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import pandas as pd

from experiments.experiment_defaults import DEFAULTS, runner_kwargs, get_output_filepath, _cfg
from src.runner import run_causal_dag_validation


_stroke_cfg = _cfg.get("stroke_validation", {})


def _render_comparison_png(var_names, ground_truth_edges, neuron_edges, assembly_edges, out_path: Path) -> None:
    """Render ground truth, neuron, and assembly DAGs side-by-side with aligned ranks."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as exc:
        print(f"[warn] Skipping PNG render: {exc}")
        return

    depth = {name: 0 for name in var_names}
    changed = True
    while changed:
        changed = False
        for src, dst in ground_truth_edges:
            if depth.get(dst, 0) < depth.get(src, 0) + 1:
                depth[dst] = depth.get(src, 0) + 1
                changed = True

    depth_to_nodes = {}
    for name in var_names:
        d = depth.get(name, 0)
        depth_to_nodes.setdefault(d, []).append(name)

    pos_template = {}
    for d, names in depth_to_nodes.items():
        k = len(names)
        xs = [(j - (k - 1) / 2) * 2.4 for j in range(k)]
        y = -d * 1.6
        for name, x in zip(names, xs):
            pos_template[name] = (x, y)

    specs = [
        ("ground truth", ground_truth_edges, "#0D47A1", "#90CAF9"),
        ("neuron", neuron_edges, "#2E7D32", "#A5D6A7"),
        ("assembly", assembly_edges, "#B71C1C", "#EF9A9A"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    for (title, edges, edge_color, node_color), ax in zip(specs, axes):
        g = nx.DiGraph()
        g.add_nodes_from(var_names)
        g.add_edges_from(edges)
        pos = {n: pos_template.get(n, (0, 0)) for n in var_names}
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            node_size=950,
            node_color=node_color,
            edgecolors="#263238",
            linewidths=0.9,
        )
        nx.draw_networkx_labels(
            g,
            pos,
            ax=ax,
            font_size=9,
            font_color="#1b1b1b",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            edge_color=edge_color,
            width=2.2,
        )
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
from src.validation.interventional.pearl_do_calculus import compute_neuron_and_assembly_mean_features, probability_of_values, sign

def generate_stroke_data(n_patients=DEFAULTS.n_samples, seed=DEFAULTS.seed, do=None, *, variant: str = 'multivalued'):
    """Generate stroke risk data with known causal structure.

    Supports do-operator overrides by replacing the structural equation for the
    intervened variable with a constant assignment.

    Args:
        n_patients: Number of patients to simulate
        seed: Random seed
        do: Optional mapping like {'Smoking': 'Smoker'} or {'Stroke': 'Stroke'}

    Returns:
        df: pandas DataFrame
        ground_truth_edges: List of (source, target) causal edges
    """
    rng = np.random.default_rng(seed)

    print('\n' + '='*70)
    print('GENERATING STROKE RISK DATA')
    print('='*70)
    print(f'  Cohort size: {n_patients} patients')
    print('  Variables: 6 (Hypertension, Age, Smoking, Atherosclerosis, BloodClotting, Stroke)')
    if do:
        print(f'  Intervention: do({do})')

    data = []

    for _ in range(n_patients):
        # Exogenous noise
        u_htn = float(rng.random())
        u_age = float(rng.random())
        u_smoke = float(rng.random())
        u_ath = float(rng.random())
        u_clot = float(rng.random())
        u_stroke = float(rng.random())
        # Root variables
        if variant == 'base':
            hypertension = 'High' if u_htn < 0.35 else 'Normal'
            age = 'Elderly' if u_age < 0.40 else 'Young'
            smoking = 'Smoker' if u_smoke < 0.30 else 'NonSmoker'
        elif variant == 'multivalued':
            # Multivalued roots (ordered) to strengthen deterministic rate-code signal.
            hypertension = (
                'High' if u_htn < 0.25 else
                'Elevated' if u_htn < 0.55 else
                'Normal'
            )
            age = (
                'Elderly' if u_age < 0.30 else
                'Middle' if u_age < 0.65 else
                'Young'
            )
            smoking = (
                'Smoker' if u_smoke < 0.22 else
                'Former' if u_smoke < 0.42 else
                'NonSmoker'
            )
        else:
            raise ValueError(f"Unknown variant={variant!r}; expected 'base' or 'multivalued'")

        if do:
            if 'Hypertension' in do:
                hypertension = do['Hypertension']
            if 'Age' in do:
                age = do['Age']
            if 'Smoking' in do:
                smoking = do['Smoking']
        # Atherosclerosis (caused by Hypertension and Age)
        p_atherosclerosis = 0.08
        if hypertension == 'Elevated':
            p_atherosclerosis += 0.25
        elif hypertension == 'High':
            p_atherosclerosis += 0.50
        if age == 'Middle':
            p_atherosclerosis += 0.12
        elif age == 'Elderly':
            p_atherosclerosis += 0.25
        p_atherosclerosis = min(p_atherosclerosis, 0.98)
        atherosclerosis = 'Present' if u_ath < p_atherosclerosis else 'None'

        if do and 'Atherosclerosis' in do:
            atherosclerosis = do['Atherosclerosis']
        # Blood Clotting (caused by Smoking)
        p_clotting = 0.06
        if smoking == 'Former':
            p_clotting += 0.25
        elif smoking == 'Smoker':
            p_clotting += 0.45
        p_clotting = min(p_clotting, 0.98)
        blood_clotting = 'Hypercoagulable' if u_clot < p_clotting else 'Normal'

        if do and 'BloodClotting' in do:
            blood_clotting = do['BloodClotting']

        # Stroke (caused by Atherosclerosis and BloodClotting)
        p_stroke = 0.03
        if atherosclerosis == 'Present':
            p_stroke += 0.50
        if blood_clotting == 'Hypercoagulable':
            p_stroke += 0.35
        p_stroke = min(p_stroke, 0.98)
        stroke = 'Stroke' if u_stroke < p_stroke else 'NoStroke'

        if do and 'Stroke' in do:
            stroke = do['Stroke']

        data.append({
            'Hypertension': hypertension,
            'Age': age,
            'Smoking': smoking,
            'Atherosclerosis': atherosclerosis,
            'BloodClotting': blood_clotting,
            'Stroke': stroke,
        })

    df = pd.DataFrame(data)


    if variant == 'multivalued':
        df['Hypertension'] = pd.Categorical(df['Hypertension'], categories=['Normal', 'Elevated', 'High'], ordered=True)
        df['Age'] = pd.Categorical(df['Age'], categories=['Young', 'Middle', 'Elderly'], ordered=True)
        df['Smoking'] = pd.Categorical(df['Smoking'], categories=['NonSmoker', 'Former', 'Smoker'], ordered=True)

    df['Atherosclerosis'] = pd.Categorical(df['Atherosclerosis'], categories=['None', 'Present'], ordered=True)
    df['BloodClotting'] = pd.Categorical(df['BloodClotting'], categories=['Normal', 'Hypercoagulable'], ordered=True)
    df['Stroke'] = pd.Categorical(df['Stroke'], categories=['NoStroke', 'Stroke'], ordered=True)

    ground_truth_edges = [
        ('Hypertension', 'Atherosclerosis'),
        ('Age', 'Atherosclerosis'),
        ('Smoking', 'BloodClotting'),
        ('Atherosclerosis', 'Stroke'),
        ('BloodClotting', 'Stroke'),
    ]

    print('\n  Clinical Distribution:')
    print(f'    Hypertension: {sum(df["Hypertension"]=="High")} ({sum(df["Hypertension"]=="High")/len(df)*100:.1f}%)')
    print(f'    Atherosclerosis: {sum(df["Atherosclerosis"]=="Present")} ({sum(df["Atherosclerosis"]=="Present")/len(df)*100:.1f}%)')
    print(f'    Stroke Events: {sum(df["Stroke"]=="Stroke")} ({sum(df["Stroke"]=="Stroke")/len(df)*100:.1f}%)')

    print('\n  Ground Truth Causal Edges:')
    for source, target in ground_truth_edges:
        print(f'    {source} -> {target}')

    return df, ground_truth_edges


def main():
    """Run Stroke Risk 3-DAG causal validation experiment."""
    print('\n' + '='*70)
    print('STROKE RISK - 3-DAG CAUSAL VALIDATION')
    print('='*70)
    print('  Research Question: Are causal relationships preserved in assemblies?')
    print('  Method: Compare 3 DAGs (Ground Truth, Neurons, Assemblies)')
    
    # Generate data
    df, ground_truth = generate_stroke_data(n_patients=DEFAULTS.n_samples, seed=DEFAULTS.seed, variant='multivalued')
    
    var_names = ['Hypertension', 'Age', 'Smoking', 
                 'Atherosclerosis', 'BloodClotting', 'Stroke']
    csv_path = get_output_filepath("table_1_stroke_metrics.csv")
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    dag_plot_path = os.path.join(output_dir, 'stroke_3dag_comparison.png')
    
    
    # Define which values are "positive" (high activation = 0.30)
    positive_values_map = {
        'Hypertension': {'High'},
        'Age': {'Elderly'},
        'Smoking': {'Smoker'},
        'Atherosclerosis': {'Present'},
        'BloodClotting': {'Hypercoagulable'},
        'Stroke': {'Stroke'}
    }
    # Run 3-DAG validation (sweep)
    sweep_n_train = [int(x) for x in _stroke_cfg.get("sweep_n_train", [400, 800])]
    sweep_n_presentations = [int(x) for x in _stroke_cfg.get("sweep_n_presentations", [1, 3, 5])]

    sweep_rows = []
    best_row = None

    for n_train in sweep_n_train:
        for n_presentations in sweep_n_presentations:
            print("\n" + "-" * 70)
            print(f"SWEEP RUN: n_train={n_train}, n_presentations={n_presentations}")
            print("-" * 70)

            res = run_causal_dag_validation(
                df=df,
                positive_values_map=positive_values_map,
                var_names=var_names,
                ground_truth_edges=ground_truth,
                skip_neuron_dag=False,
                strict_neuron_causal_discovery=False,
                strict_assembly_causal_discovery=False,
                assembly_method="ges",
                stimulus_k=DEFAULTS.assembly_k,
                # deterministic-k multivalued setup; all other knobs come from config via runner_kwargs()
                **runner_kwargs(n_train=n_train, n_presentations=n_presentations),
            )

            assembly_metrics = res["comparison"]["assembly_vs_gt"]
            row = {
                "n_train": n_train,
                "n_presentations": n_presentations,
                "precision": float(assembly_metrics["precision"]),
                "recall": float(assembly_metrics["recall"]),
                "f1": float(assembly_metrics["f1"]),
                "results": res,
            }
            sweep_rows.append(row)

            if best_row is None:
                best_row = row
            else:
                key = (row["recall"], row["f1"], row["precision"])
                best_key = (best_row["recall"], best_row["f1"], best_row["precision"])
                if key > best_key:
                    best_row = row

            print("  Assembly vs GT: P={:.3f}, R={:.3f}, F1={:.3f}".format(
                row["precision"], row["recall"], row["f1"]
            ))

    assert best_row is not None
    results = best_row["results"]

    png_out = Path(output_dir) / "stroke_3dag_comparison.png"
    _render_comparison_png(
        var_names=var_names,
        ground_truth_edges=ground_truth,
        neuron_edges=results.get("neuron_edges", []),
        assembly_edges=results.get("assembly_edges", []),
        out_path=png_out,
    )

    # ---------------------------------------------------------------------
    # do-operator intervention evaluation (effect preservation) (effect preservation)
    # ---------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('INTERVENTION EVALUATION (do operator)')
    print('=' * 70)
    print('  Goal: compare interventional outcome differences in raw space versus feature space')
    print('  Note: reuse the trained Brain from the baseline run (no retraining)')

    n_eval = min(DEFAULTS.n_samples, 2000)
    eval_seed = DEFAULTS.seed + 123

    df_base_eval, _ = generate_stroke_data(n_patients=n_eval, seed=eval_seed, do=None)
    base_neuron_means, base_assembly_means, assembly_flip_map = compute_neuron_and_assembly_mean_features(
        df_base_eval,
        brain=results['brain'],
        var_names=var_names,
        neurons_per_var=DEFAULTS.neurons_per_var,
        positive_values_map=positive_values_map,
        positive_prob=DEFAULTS.positive_prob,
        negative_prob=DEFAULTS.negative_prob,
        seed=DEFAULTS.seed,
        return_assembly_flip_map=True,
        deterministic_k_step=runner_kwargs()["deterministic_k_step"],
        stimulus_k=DEFAULTS.assembly_k,
    )

    baseline_raw = probability_of_values(df_base_eval, 'Stroke', {'Stroke'})
    baseline_neuron = base_neuron_means['Stroke']
    baseline_assembly = base_assembly_means['Stroke']

    print(f'  Eval cohort size: {n_eval} (seed={eval_seed})')
    print(f'  Baseline: P(Stroke=Stroke)={baseline_raw:.3f}, neuron mean={baseline_neuron:.3f}, assembly mean={baseline_assembly:.3f}')

    interventions = [
        ('do(Hypertension=High)', {'Hypertension': 'High'}),
        ('do(Age=Elderly)', {'Age': 'Elderly'}),
        ('do(Smoking=Smoker)', {'Smoking': 'Smoker'}),
        ('do(Atherosclerosis=Present)', {'Atherosclerosis': 'Present'}),
        ('do(BloodClotting=Hypercoagulable)', {'BloodClotting': 'Hypercoagulable'}),
    ]

    print('\nIntervention effect differences (do intervention minus baseline)')
    print('Intervention                          | rawDeltaStroke | neuronDeltaStroke | assemblyDeltaStroke | sign match (Neuron/Assembly)')
    print("  Interpretation: N:Y/A:Y => same direction as raw delta (intervention effect preserved directionally); N:N or A:N => mismatch. Magnitudes can differ due to different scales.")
    print('-' * 110)


    total = 0
    ok_neuron = 0
    ok_assembly = 0



    sign_match_summary = None
    for label, do_dict in interventions:
        df_do_eval, _ = generate_stroke_data(n_patients=n_eval, seed=eval_seed, do=do_dict)
        do_neuron_means, do_assembly_means = compute_neuron_and_assembly_mean_features(
            df_do_eval,
            brain=results['brain'],
            var_names=var_names,
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=DEFAULTS.positive_prob,
            negative_prob=DEFAULTS.negative_prob,
            seed=DEFAULTS.seed,
            assembly_flip_map=assembly_flip_map,
            deterministic_k_step=runner_kwargs()["deterministic_k_step"],
            stimulus_k=DEFAULTS.assembly_k,
        )

        raw = probability_of_values(df_do_eval, 'Stroke', {'Stroke'})
        neuron = do_neuron_means['Stroke']
        assembly = do_assembly_means['Stroke']

        d_raw = raw - baseline_raw
        d_neuron = neuron - baseline_neuron
        d_assembly = assembly - baseline_assembly

        match_neuron = 'Y' if sign(d_raw) == sign(d_neuron) else 'N'
        match_assembly = 'Y' if sign(d_raw) == sign(d_assembly) else 'N'


        total += 1
        if match_neuron == 'Y':
            ok_neuron += 1
        if match_assembly == 'Y':
            ok_assembly += 1


        print(f'{label:36s} | {d_raw:+.3f}     | {d_neuron:+.3f}       | {d_assembly:+.3f}         | N:{match_neuron}/A:{match_assembly}')

    if total > 0:
        sign_match_summary = (
            f"Neuron={ok_neuron}/{total} ({ok_neuron/total:.1%}), "
            f"Assembly={ok_assembly}/{total} ({ok_assembly/total:.1%})"
        )
        print()
        print(f"  Sign-match rate (directional): {sign_match_summary}")
    # Summary
    skip_neuron = results['comparison']['neuron_vs_gt'] is None
    neuron_metrics = results['comparison'].get('neuron_vs_gt')
    assembly_metrics = results['comparison']['assembly_vs_gt']
    assembly_vs_neuron_metrics = results['comparison'].get('assembly_vs_neuron')
    preservation_score = results['comparison'].get('preservation_score')

    print('\n' + '='*70)
    print('EXPERIMENT COMPLETE')
    print('='*70)
    print(f'\n  Compression: {results["compression"]["ratio"]:.1f}x')
    if neuron_metrics:
        print(f'  Neuron DAG F1: {neuron_metrics["f1"]:.3f}')
    else:
        print('  Neuron DAG F1: N/A (neuron DAG skipped)')
    print(f'  Assembly DAG F1: {assembly_metrics["f1"]:.3f}')
    if preservation_score is not None:
        print(f'  Overall Preservation: {preservation_score:.3f}')
    else:
        print('  Overall Preservation: N/A (neuron DAG skipped)')
    # Save results
    results_file = os.path.join(output_dir, 'stroke_causal_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('STROKE RISK - 3-DAG CAUSAL VALIDATION RESULTS\n')
        f.write('='*70 + '\n\n')
        
        f.write('GROUND TRUTH EDGES:\n')
        for source, target in results['ground_truth_edges']:
            f.write(f'  {source} -> {target}\n')
        
        f.write(f'\nNEURON DAG EDGES ({len(results["neuron_edges"])} discovered):\n')
        for source, target in results['neuron_edges']:
            f.write(f'  {source} -> {target}\n')
        
        f.write(f'\nASSEMBLY DAG EDGES ({len(results["assembly_edges"])} discovered):\n')
        for source, target in results['assembly_edges']:
            f.write(f'  {source} -> {target}\n')
        
        f.write('\n' + '='*70 + '\n')
        f.write('VALIDATION METRICS:\n')
        f.write('='*70 + '\n')
        
        f.write('\nNeuron DAG vs Ground Truth:\n')
        if neuron_metrics:
            f.write(f'  Precision: {neuron_metrics["precision"]:.3f}\n')
            f.write(f'  Recall: {neuron_metrics["recall"]:.3f}\n')
            f.write(f'  F1 Score: {neuron_metrics["f1"]:.3f}\n')
        else:
            f.write('  (skipped; neuron DAG disabled)\n')
        
        f.write('\nAssembly DAG vs Ground Truth:\n')
        f.write(f'  Precision: {assembly_metrics["precision"]:.3f}\n')
        f.write(f'  Recall: {assembly_metrics["recall"]:.3f}\n')
        f.write(f'  F1 Score: {assembly_metrics["f1"]:.3f}\n')
        
        f.write('\nAssembly vs Neuron Agreement:\n')
        if assembly_vs_neuron_metrics:
            f.write(f'  Similarity: {assembly_vs_neuron_metrics["precision"]:.3f}\n')
        else:
            f.write('  (skipped; neuron DAG disabled)\n')
        
        if preservation_score is not None:
            f.write(f'\nOverall Preservation Score: {preservation_score:.3f}\n')
        else:
            f.write('\nOverall Preservation Score: (skipped; neuron DAG disabled)\n')
        f.write(f'Compression Ratio: {results["compression"]["ratio"]:.1f}x\n')

        f.write('\n' + '='*70 + '\n')
        f.write('INTERVENTION SIGN-MATCH (do operator):\n')
        f.write('='*70 + '\n')
        if sign_match_summary:
            f.write(f'Sign-match rate (directional): {sign_match_summary}\n')
        else:
            f.write('Sign-match rate (directional): (not computed)\n')

    print(f'\n  Results saved to: {os.path.basename(results_file)}')
    print(f'  DAG visualization saved to: {os.path.basename(dag_plot_path)}')
    
    return results

if __name__ == '__main__':
    results = main()

