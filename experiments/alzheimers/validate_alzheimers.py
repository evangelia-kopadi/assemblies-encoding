"""
Alzheimer's Disease Case Study - 3-DAG Causal Validation

Validates that neural assemblies preserve causal information by comparing:
1. Ground Truth DAG (from literature)
2. Neuron DAG (causal discovery on neuron activations)
3. Assembly DAG (causal discovery on assembly activations)

Ground Truth Causal Structure (10 variables):
  APOE4 -> Amyloid -> CognitiveDecline
  PhysicalActivity -> Amyloid (protective)
  Cholesterol -> Amyloid
  Inflammation -> Amyloid, Tau
  SleepQuality -> Tau (poor sleep worsens Tau)
  Diet -> Tau -> CognitiveDecline
  Education -> CognitiveDecline (low education increases risk)
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import pandas as pd

from experiments.experiment_defaults import DEFAULTS, runner_kwargs
from src.runner import run_causal_dag_validation
from src.visualization.dag_plotting import visualize_three_dags
from src.validation.interventional.pearl_do_calculus import compute_neuron_and_assembly_mean_features, probability_of_values, sign

def generate_alzheimers_data(n_patients=DEFAULTS.n_samples, seed=DEFAULTS.seed, do=None):
    """Generate synthetic Alzheimer data with an extended structural causal model.

    Supports do-operator overrides by replacing the structural equation for the
    intervened variable with a constant assignment.
    """
    rng = np.random.default_rng(seed)

    print('\n' + '='*70)
    print('GENERATING ALZHEIMER PATIENT DATA')
    print('='*70)
    print(f'  Cohort size: {n_patients} patients')
    print('  Variables: 10 (APOE4, PhysicalActivity, Diet, Cholesterol, Inflammation, SleepQuality, Education, Amyloid, Tau, CognitiveDecline)')
    if do:
        print(f'  Intervention: do({do})')

    data = []

    for _ in range(n_patients):
        # Exogenous noise
        u_apoe = float(rng.random())
        u_pa = float(rng.random())
        u_diet = float(rng.random())
        u_chol = float(rng.random())
        u_inf = float(rng.random())
        u_sleep = float(rng.random())
        u_edu = float(rng.random())
        u_amy = float(rng.random())
        u_tau = float(rng.random())
        u_cd = float(rng.random())

        # Root variables
        apoe4 = 'Carrier' if u_apoe < 0.30 else 'Non-carrier'
        physical_activity = 'Sedentary' if u_pa < 0.40 else 'Active'
        diet = 'Poor' if u_diet < 0.45 else 'Healthy'
        cholesterol = 'High' if u_chol < 0.35 else 'Normal'
        inflammation = 'High' if u_inf < 0.25 else 'Normal'
        sleep_quality = 'Poor' if u_sleep < 0.40 else 'Good'
        education = 'Low' if u_edu < 0.35 else 'High'

        if do:
            if 'APOE4' in do:
                apoe4 = do['APOE4']
            if 'PhysicalActivity' in do:
                physical_activity = do['PhysicalActivity']
            if 'Diet' in do:
                diet = do['Diet']
            if 'Cholesterol' in do:
                cholesterol = do['Cholesterol']
            if 'Inflammation' in do:
                inflammation = do['Inflammation']
            if 'SleepQuality' in do:
                sleep_quality = do['SleepQuality']
            if 'Education' in do:
                education = do['Education']

        # Amyloid structural equation
        p_high_amyloid = 0.15
        if apoe4 == 'Carrier':
            p_high_amyloid += 0.50
        if physical_activity == 'Sedentary':
            p_high_amyloid += 0.20
        if cholesterol == 'High':
            p_high_amyloid += 0.25
        if inflammation == 'High':
            p_high_amyloid += 0.20
        p_high_amyloid = min(p_high_amyloid, 0.98)
        amyloid = 'High' if u_amy < p_high_amyloid else 'Normal'

        if do and 'Amyloid' in do:
            amyloid = do['Amyloid']

        # Tau structural equation
        p_high_tau = 0.10
        if diet == 'Poor':
            p_high_tau += 0.35
        if inflammation == 'High':
            p_high_tau += 0.20
        if sleep_quality == 'Poor':
            p_high_tau += 0.15
        p_high_tau = min(p_high_tau, 0.98)
        tau = 'High' if u_tau < p_high_tau else 'Normal'

        if do and 'Tau' in do:
            tau = do['Tau']

        # CognitiveDecline structural equation
        p_decline = 0.05
        if amyloid == 'High':
            p_decline += 0.45
        if tau == 'High':
            p_decline += 0.30
        if education == 'Low':
            p_decline += 0.10
        p_decline = min(p_decline, 0.98)
        cognitive_decline = 'Decline' if u_cd < p_decline else 'Stable'

        if do and 'CognitiveDecline' in do:
            cognitive_decline = do['CognitiveDecline']

        data.append({
            'APOE4': apoe4,
            'PhysicalActivity': physical_activity,
            'Diet': diet,
            'Cholesterol': cholesterol,
            'Inflammation': inflammation,
            'SleepQuality': sleep_quality,
            'Education': education,
            'Amyloid': amyloid,
            'Tau': tau,
            'CognitiveDecline': cognitive_decline,
        })

    df = pd.DataFrame(data)

    ground_truth_edges = [
        ('APOE4', 'Amyloid'),
        ('PhysicalActivity', 'Amyloid'),
        ('Diet', 'Tau'),
        ('Cholesterol', 'Amyloid'),
        ('Inflammation', 'Amyloid'),
        ('Inflammation', 'Tau'),
        ('SleepQuality', 'Tau'),
        ('Amyloid', 'CognitiveDecline'),
        ('Tau', 'CognitiveDecline'),
        ('Education', 'CognitiveDecline'),
    ]

    print('\n  Clinical Distribution:')
    print(f'    APOE4 Carriers: {sum(df["APOE4"]=="Carrier")} ({sum(df["APOE4"]=="Carrier")/len(df)*100:.1f}%)')
    print(f'    High Cholesterol: {sum(df["Cholesterol"]=="High")} ({sum(df["Cholesterol"]=="High")/len(df)*100:.1f}%)')
    print(f'    High Inflammation: {sum(df["Inflammation"]=="High")} ({sum(df["Inflammation"]=="High")/len(df)*100:.1f}%)')
    print(f'    Poor Sleep: {sum(df["SleepQuality"]=="Poor")} ({sum(df["SleepQuality"]=="Poor")/len(df)*100:.1f}%)')
    print(f'    Low Education: {sum(df["Education"]=="Low")} ({sum(df["Education"]=="Low")/len(df)*100:.1f}%)')
    print(f'    High Amyloid: {sum(df["Amyloid"]=="High")} ({sum(df["Amyloid"]=="High")/len(df)*100:.1f}%)')
    print(f'    Cognitive Decline: {sum(df["CognitiveDecline"]=="Decline")} ({sum(df["CognitiveDecline"]=="Decline")/len(df)*100:.1f}%)')

    print('\n  Ground Truth Causal Edges:')
    for source, target in ground_truth_edges:
        print(f'    {source} -> {target}')

    return df, ground_truth_edges

def main():
    """Run Alzheimer's 3-DAG causal validation experiment."""
    print('\n' + '='*70)
    print('ALZHEIMER DISEASE - 3-DAG CAUSAL VALIDATION')
    print('='*70)
    print('  Research Question: Are causal relationships preserved in assemblies?')
    print('  Method: Compare 3 DAGs (Ground Truth, Neurons, Assemblies)')
    
    # Generate data
    df, ground_truth = generate_alzheimers_data(n_patients=DEFAULTS.n_samples, seed=DEFAULTS.seed)
    
    var_names = [
        'APOE4', 'PhysicalActivity', 'Diet', 'Cholesterol', 'Inflammation', 'SleepQuality', 'Education',
        'Amyloid', 'Tau', 'CognitiveDecline'
    ]

    positive_values_map = {
        'APOE4': {'Carrier'},
        'PhysicalActivity': {'Sedentary'},
        'Diet': {'Poor'},
        'Cholesterol': {'High'},
        'Inflammation': {'High'},
        'SleepQuality': {'Poor'},
        'Education': {'Low'},
        'Amyloid': {'High'},
        'Tau': {'High'},
        'CognitiveDecline': {'Decline'},
    }
    
    # Run 3-DAG validation
    results = run_causal_dag_validation(
        df=df,
        var_names=var_names,
        ground_truth_edges=ground_truth,
        skip_neuron_dag=False,
        jitter_std=0.00,
        **runner_kwargs()
    )
    
    # ---------------------------------------------------------------------
    # do-operator intervention evaluation (effect preservation)
    # ---------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('INTERVENTION EVALUATION (do operator)')
    print('=' * 70)
    print('  Goal: compare interventional outcome differences in raw space versus feature space')
    print('  Note: reuse the trained Brain from the baseline run (no retraining)')

    n_eval = min(DEFAULTS.n_samples, 2000)
    eval_seed = DEFAULTS.seed + 123

    df_base_eval, _ = generate_alzheimers_data(n_patients=n_eval, seed=eval_seed, do=None)
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
    )

    baseline_prob_by_var = {
        var: probability_of_values(df_base_eval, var, positive_values_map.get(var, set()))
        for var in var_names
    }
    baseline_raw = baseline_prob_by_var['CognitiveDecline']
    baseline_neuron = base_neuron_means['CognitiveDecline']
    baseline_assembly = base_assembly_means['CognitiveDecline']

    print(f'  Eval cohort size: {n_eval} (seed={eval_seed})')
    print(f'  Baseline: P(CognitiveDecline=Decline)={baseline_raw:.3f}, neuron mean={baseline_neuron:.3f}, assembly mean={baseline_assembly:.3f}')

    interventions = [
        ('do(APOE4=Carrier)', {'APOE4': 'Carrier'}),
        ('do(PhysicalActivity=Sedentary)', {'PhysicalActivity': 'Sedentary'}),
        ('do(Diet=Poor)', {'Diet': 'Poor'}),
        ('do(Cholesterol=High)', {'Cholesterol': 'High'}),
        ('do(Inflammation=High)', {'Inflammation': 'High'}),
        ('do(SleepQuality=Poor)', {'SleepQuality': 'Poor'}),
        ('do(Education=Low)', {'Education': 'Low'}),
        ('do(Amyloid=High)', {'Amyloid': 'High'}),
        ('do(Tau=High)', {'Tau': 'High'}),
    ]

    print('\nIntervention effect differences (do intervention minus baseline)')
    print('Intervention                          | rawDeltaDecline | neuronDeltaDecline | assemblyDeltaDecline | sign match (Neuron/Assembly)')
    print("  Interpretation: N:Y/A:Y => same direction as raw delta (intervention effect preserved directionally); N:N or A:N => mismatch. Magnitudes can differ due to different scales.")
    print('-' * 110)


    total = 0
    ok_neuron = 0
    ok_assembly = 0

    sign_match_summary = None
    for label, do_dict in interventions:
        df_do_eval, _ = generate_alzheimers_data(n_patients=n_eval, seed=eval_seed, do=do_dict)
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
        )

        raw = probability_of_values(df_do_eval, 'CognitiveDecline', {'Decline'})
        neuron = do_neuron_means['CognitiveDecline']
        assembly = do_assembly_means['CognitiveDecline']

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


        print(f'{label:36s} | {d_raw:+.3f}      | {d_neuron:+.3f}        | {d_assembly:+.3f}          | N:{match_neuron}/A:{match_assembly}')

    edge_effects = []

    if total > 0:
        sign_match_summary = (
            f"Neuron={ok_neuron}/{total} ({ok_neuron/total:.1%}), "
            f"Assembly={ok_assembly}/{total} ({ok_assembly/total:.1%})"
        )
        print()
        print(f"  Sign-match rate (directional): {sign_match_summary}")

    # ------------------------------------------------------------------
    # Edge-wise ACE-style deltas: do(source=high-risk) -> target response
    # ------------------------------------------------------------------
    print('\nACE-style edge-wise deltas (do on source -> target response)')
    print('Edge (do value)                      | rawDeltaTarget | neuronDeltaTarget | assemblyDeltaTarget | sign match (Neuron/Assembly)')
    print('-' * 118)

    for source, target in ground_truth:
        source_pos = positive_values_map.get(source)
        target_pos = positive_values_map.get(target)

        if not source_pos or not target_pos:
            continue

        do_value = sorted(source_pos)[0]
        do_dict = {source: do_value}

        df_do_edge, _ = generate_alzheimers_data(n_patients=n_eval, seed=eval_seed, do=do_dict)
        do_neuron_means, do_assembly_means = compute_neuron_and_assembly_mean_features(
            df_do_edge,
            brain=results['brain'],
            var_names=var_names,
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=DEFAULTS.positive_prob,
            negative_prob=DEFAULTS.negative_prob,
            seed=DEFAULTS.seed,
            assembly_flip_map=assembly_flip_map,
        )

        raw_target = probability_of_values(df_do_edge, target, target_pos)
        d_raw = raw_target - baseline_prob_by_var[target]
        d_neuron = do_neuron_means[target] - base_neuron_means[target]
        d_assembly = do_assembly_means[target] - base_assembly_means[target]

        match_neuron = 'Y' if sign(d_raw) == sign(d_neuron) else 'N'
        match_assembly = 'Y' if sign(d_raw) == sign(d_assembly) else 'N'

        edge_effects.append({
            'source': source,
            'target': target,
            'do_value': do_value,
            'delta_raw': d_raw,
            'delta_neuron': d_neuron,
            'delta_assembly': d_assembly,
            'sign_neuron': match_neuron,
            'sign_assembly': match_assembly,
        })

        print(f"{source}->{target} ({do_value:10s})          | {d_raw:+.3f}          | {d_neuron:+.3f}            | {d_assembly:+.3f}              | N:{match_neuron}/A:{match_assembly}")

    # Visualize the 3 DAGs
    output_dir = os.path.dirname(__file__)
    dag_plot_path = os.path.join(output_dir, 'alzheimers_3dag_comparison.png')
    
    visualize_three_dags(
        ground_truth_edges=results['ground_truth_edges'],
        neuron_edges=results['neuron_edges'],
        assembly_edges=results['assembly_edges'],
        var_names=var_names,
        save_path=dag_plot_path
    )
    
    # Summary
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
    results_file = os.path.join(output_dir, 'alzheimers_causal_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('ALZHEIMER DISEASE - 3-DAG CAUSAL VALIDATION RESULTS\n')
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

        f.write('\nACE-style edge-wise deltas (do on source -> target response):\n')
        if edge_effects:
            f.write('  Source->Target (do value) | rawDelta | neuronDelta | assemblyDelta | sign match (Neuron/Assembly)\n')
            for row in edge_effects:
                f.write(
                    f"  {row['source']}->{row['target']} ({row['do_value']}) | "
                    f"{row['delta_raw']:+.3f} | {row['delta_neuron']:+.3f} | {row['delta_assembly']:+.3f} | "
                    f"N:{row['sign_neuron']}/A:{row['sign_assembly']}\n"
                )
        else:
            f.write('  (not computed)\n')
    
    print(f'\n  Results saved to: {os.path.basename(results_file)}')
    print(f'  DAG visualization saved to: {os.path.basename(dag_plot_path)}')
    
    return results

if __name__ == '__main__':
    results = main()



