"""
Vaccine Efficacy - EXTREMELY Strong Causal Effects
Medical domain with near-deterministic relationships
"""

import sys
import json
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import os
from experiments.experiment_defaults import DEFAULTS, runner_kwargs, get_output_filepath
from src.runner import run_causal_dag_validation
from src.visualization.dag_plotting import visualize_three_dags
from src.validation.interventional.pearl_do_calculus import compute_neuron_and_assembly_mean_features, probability_of_values, sign

def generate_vaccine_data(n_patients=DEFAULTS.n_samples, seed=DEFAULTS.seed, do=None):
    """
    Generate vaccine data with VERY STRONG causal effects.
    
    Ground Truth:
      Vaccination -> AntibodyResponse -> Immunity
      ImmuneHealth -> AntibodyResponse (co-factor)
    """
    np.random.seed(seed)
    
    print('\n' + '='*70)
    print('GENERATING VACCINE EFFICACY DATA')
    print('='*70)
    print(f'  Cohort size: {n_patients} patients')
    if do:
        print(f'  Intervention: do({do})')
    
    data = []
    
    for _ in range(n_patients):
        # Baseline characteristics
        vaccination = np.random.choice(['Unvaccinated', 'Vaccinated'], p=[0.30, 0.70])
        immune_health = np.random.choice(['Weak', 'Strong'], p=[0.25, 0.75])

        if do:
            if 'Vaccination' in do:
                vaccination = do['Vaccination']
            if 'ImmuneHealth' in do:
                immune_health = do['ImmuneHealth']
        
        # Antibody Response - VERY STRONG effects (near deterministic)
        p_antibodies = 0.02  # Almost zero baseline (unvaccinated rarely produce antibodies)
        if vaccination == 'Vaccinated':
            p_antibodies += 0.85  # VERY STRONG (vaccination → antibodies is highly reliable)
        if immune_health == 'Strong':
            p_antibodies += 0.10  # Boosts response
        
        # Cap at 0.98 to avoid determinism
        p_antibodies = min(p_antibodies, 0.98)
        
        antibody_response = np.random.choice(['Low', 'High'], 
                                            p=[1 - p_antibodies, p_antibodies])

        if do and 'AntibodyResponse' in do:
            antibody_response = do['AntibodyResponse']
        
        # Immunity - VERY STRONG effect from antibodies
        p_immunity = 0.05  # Low baseline (natural immunity is rare)
        if antibody_response == 'High':
            p_immunity += 0.85  # VERY STRONG (high antibodies  immunity)
        
        immunity = np.random.choice(['NotImmune', 'Immune'], 
                                   p=[1 - p_immunity, p_immunity])

        if do and 'Immunity' in do:
            immunity = do['Immunity']
        
        data.append({
            'Vaccination': vaccination,
            'ImmuneHealth': immune_health,
            'AntibodyResponse': antibody_response,
            'Immunity': immunity
        })
    
    df = pd.DataFrame(data)
    
    ground_truth_edges = [
        ('Vaccination', 'AntibodyResponse'),
        ('ImmuneHealth', 'AntibodyResponse'),
        ('AntibodyResponse', 'Immunity')
    ]
    
    print('\nGround Truth Causal Structure:')
    for source, target in ground_truth_edges:
        print(f'  {source} -> {target}')
    
    print('\nOutcome Distribution:')
    print(f'  Vaccinated: {(df["Vaccination"] == "Vaccinated").sum()} ({(df["Vaccination"] == "Vaccinated").mean() * 100:.1f}%)')
    print(f'  High Antibodies: {(df["AntibodyResponse"] == "High").sum()} ({(df["AntibodyResponse"] == "High").mean() * 100:.1f}%)')
    print(f'  Immune: {(df["Immunity"] == "Immune").sum()} ({(df["Immunity"] == "Immune").mean() * 100:.1f}%)')
    
    return df, ground_truth_edges

if __name__ == '__main__':
    # Prepare output file
    csv_path = get_output_filepath("table_1_vaccine_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df, ground_truth_edges = generate_vaccine_data(n_patients=DEFAULTS.n_samples, seed=DEFAULTS.seed)
    
    # Define positive values for correct encoding
    positive_values_map = {
        'Vaccination': {'Vaccinated'},
        'ImmuneHealth': {'Strong'},
        'AntibodyResponse': {'High'},
        'Immunity': {'Immune'}
    }
    
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
    print('\n' + '=' * 70)
    print('INTERVENTION EVALUATION (do operator)')
    print('=' * 70)
    print('  Goal: compare interventional outcome differences in raw space versus feature space')
    print('  Note: reuse the trained Brain from the baseline run (no retraining)')

    n_eval = min(DEFAULTS.n_samples, 2000)
    eval_seed = DEFAULTS.seed + 123

    df_base_eval, _ = generate_vaccine_data(n_patients=n_eval, seed=eval_seed, do=None)
    var_names = list(df_base_eval.columns)

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

    baseline_raw = probability_of_values(df_base_eval, 'Immunity', {'Immune'})
    baseline_neuron = base_neuron_means['Immunity']
    baseline_assembly = base_assembly_means['Immunity']

    print(f'  Eval cohort size: {n_eval} (seed={eval_seed})')
    print(f'  Baseline: P(Immunity=Immune)={baseline_raw:.3f}, neuron mean={baseline_neuron:.3f}, assembly mean={baseline_assembly:.3f}')

    interventions = [
        ('do(Vaccination=Vaccinated)', {'Vaccination': 'Vaccinated'}),
        ('do(Vaccination=Unvaccinated)', {'Vaccination': 'Unvaccinated'}),
        ('do(ImmuneHealth=Strong)', {'ImmuneHealth': 'Strong'}),
        ('do(AntibodyResponse=High)', {'AntibodyResponse': 'High'}),
    ]

    print('\nIntervention effect differences (do intervention minus baseline)')
    print('Intervention                          | raw change | neuron change | assembly change | sign match (Neuron/Assembly)')
    print("  Interpretation: N:Y/A:Y => same direction as raw change (intervention effect preserved directionally); N:N or A:N => mismatch. Magnitudes can differ due to different scales.")
    print('-' * 110)


    total = 0
    ok_neuron = 0
    ok_assembly = 0



    sign_match_summary = None
    for label, do_dict in interventions:
        df_do_eval, _ = generate_vaccine_data(n_patients=n_eval, seed=eval_seed, do=do_dict)
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

        raw = probability_of_values(df_do_eval, 'Immunity', {'Immune'})
        neuron = do_neuron_means['Immunity']
        assembly = do_assembly_means['Immunity']

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


        print(f'{label:36s} | {d_raw:+.3f}      | {d_neuron:+.3f}       | {d_assembly:+.3f}        | N:{match_neuron}/A:{match_assembly}')

    if total > 0:
        sign_match_summary = (
            f"Neuron={ok_neuron}/{total} ({ok_neuron/total:.1%}), "
            f"Assembly={ok_assembly}/{total} ({ok_assembly/total:.1%})"
        )
        print()
        print(f"  Sign-match rate (directional): {sign_match_summary}")
    visualize_three_dags(
        ground_truth_edges=ground_truth_edges,
        neuron_edges=results['neuron_edges'],
        assembly_edges=results['assembly_edges'],
        var_names=list(df.columns),
        save_path=os.path.join(os.path.dirname(csv_path), 'vaccine_3dag_comparison.png')
    )
    
    # Save detailed results
    with open(os.path.join(os.path.dirname(csv_path), 'vaccine_causal_results.txt'), 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('VACCINE EFFICACY - 3-DAG CAUSAL VALIDATION RESULTS\n')
        f.write('='*70 + '\n\n')
        
        f.write('GROUND TRUTH EDGES:\n')
        for source, target in ground_truth_edges:
            f.write(f'  {source} -> {target}\n')
        
        f.write(f'\nNEURON DAG EDGES ({len(results["neuron_edges"])} discovered):\n')
        for source, target in results['neuron_edges']:
            f.write(f'  {source} -> {target}\n')
        
        f.write(f'\nASSEMBLY DAG EDGES ({len(results["assembly_edges"])} discovered):\n')
        for source, target in results['assembly_edges']:
            f.write(f'  {source} -> {target}\n')
        
        f.write('\n' + '='*70 + '\n')
        f.write('VALIDATION METRICS:\n')
        f.write('='*70 + '\n\n')
        
        f.write('Neuron DAG vs Ground Truth:\n')
        f.write(f'  Precision: {results["comparison"]["neuron_vs_gt"]["precision"]:.3f}\n')
        f.write(f'  Recall: {results["comparison"]["neuron_vs_gt"]["recall"]:.3f}\n')
        f.write(f'  F1 Score: {results["comparison"]["neuron_vs_gt"]["f1"]:.3f}\n\n')
        
        f.write('Assembly DAG vs Ground Truth:\n')
        f.write(f'  Precision: {results["comparison"]["assembly_vs_gt"]["precision"]:.3f}\n')
        f.write(f'  Recall: {results["comparison"]["assembly_vs_gt"]["recall"]:.3f}\n')
        f.write(f'  F1 Score: {results["comparison"]["assembly_vs_gt"]["f1"]:.3f}\n\n')
        
        f.write('Assembly vs Neuron Agreement:\n')
        f.write(f'  Similarity: {results["comparison"]["assembly_vs_neuron"]["f1"]:.3f}\n\n')
        
        f.write(f'Overall Preservation Score: {results["comparison"]["preservation_score"]:.3f}\n')
        f.write(f'Compression Ratio: {results["compression"]["ratio"]:.1f}x\n')

        f.write('\n' + '='*70 + '\n')
        f.write('INTERVENTION SIGN-MATCH (do operator):\n')
        f.write('='*70 + '\n')
        if sign_match_summary:
            f.write(f'Sign-match rate (directional): {sign_match_summary}\n')
        else:
            f.write('Sign-match rate (directional): (not computed)\n')
    
    print('\n' + '='*70)
    print('RESULTS')
    print('='*70)
    print(f'Neuron F1: {results["comparison"]["neuron_vs_gt"]["f1"]:.3f}')
    print(f'Assembly F1: {results["comparison"]["assembly_vs_gt"]["f1"]:.3f}')











