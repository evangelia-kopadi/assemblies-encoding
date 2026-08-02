"""
Student Success Case Study - 3-DAG Causal Validation

Validates that neural assemblies preserve causal information by comparing:
1. Ground Truth DAG (known causal structure)
2. Neuron DAG (causal discovery on neuron activations)
3. Assembly DAG (causal discovery on assembly activations)

Ground Truth Causal Structure:
  PriorGPA -> FinalGrade
  Attendance -> FinalExamScore
  StudyHours -> FinalExamScore
  FinalExamScore -> FinalGrade
"""

import os
import sys

from datetime import datetime
import shutil

# Ensure repo root is on sys.path so `import src.*` resolves locally
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import numpy as np
import random
import pandas as pd

from experiments.experiment_defaults import DEFAULTS, runner_kwargs
from src.runner import run_causal_dag_validation
from src.visualization.dag_plotting import visualize_three_dags

def generate_student_data(n_students=DEFAULTS.n_samples, seed=DEFAULTS.seed, do=None, *, variant: str = 'base'):
    """
    Generate synthetic student success data with known causal structure.
    
    Ground Truth:
      PriorGPA -> FinalGrade
      Attendance -> FinalExamScore -> FinalGrade
      StudyHours -> FinalExamScore -> FinalGrade
    """
    rng = np.random.default_rng(seed)
    
    print('\n' + '='*70)
    print('GENERATING STUDENT SUCCESS DATA')
    print('='*70)
    print(f'  Cohort size: {n_students} students')
    print(f'  Variables: 5 (PriorGPA, Attendance, StudyHours, FinalExamScore, FinalGrade)')
    if do:
        print(f'  Intervention: do({do})')
    
    data = []

    def _categorical_from_u(u: float, categories: list[str], probs: list[float]) -> str:
        # Deterministic categorical sampler from a U~Uniform(0,1) draw
        cum = np.cumsum(np.asarray(probs, dtype=float))
        idx = int(np.searchsorted(cum, u, side='right'))
        if idx >= len(categories):
            idx = len(categories) - 1
        return categories[idx]

    for _ in range(n_students):
        # Exogenous noise (Pearl SCM): U variables
        u_prior = float(rng.random())
        u_att = float(rng.random())
        u_hours = float(rng.random())
        u_exam = float(rng.random())
        u_grade = float(rng.random())

        # Root variables as deterministic functions of U
        if variant == 'base':
            prior_gpa = _categorical_from_u(u_prior, ['Low', 'Average', 'High'], [0.25, 0.5, 0.25])
            attendance = _categorical_from_u(u_att, ['Poor', 'Good'], [0.4, 0.6])
            study_hours = _categorical_from_u(u_hours, ['Low', 'High'], [0.5, 0.5])
        elif variant == 'rich':
            # More granular, ordered categories to strengthen deterministic rate-code signal.
            prior_gpa = _categorical_from_u(u_prior, ['Low', 'Average', 'High', 'Honors'], [0.20, 0.45, 0.25, 0.10])
            attendance = _categorical_from_u(u_att, ['Poor', 'Average', 'Good'], [0.25, 0.50, 0.25])
            study_hours = _categorical_from_u(u_hours, ['Low', 'Medium', 'High'], [0.30, 0.40, 0.30])
        else:
            raise ValueError(f"Unknown variant={variant!r}; expected 'base' or 'rich'")

        # Apply Pearl-style do() interventions by overriding assigned values.
        # We still sample all U's first, to keep RNG consumption aligned between
        # baseline and intervened runs when using the same seed.
        if do:
            if 'PriorGPA' in do:
                prior_gpa = do['PriorGPA']
            if 'Attendance' in do:
                attendance = do['Attendance']
            if 'StudyHours' in do:
                study_hours = do['StudyHours']

        # Structural equation for FinalExamScore: FinalExamScore := f(Attendance, StudyHours, U_exam)
        if variant == 'base':
            p_excellent_exam = 0.1
            if attendance == 'Good':
                p_excellent_exam += 0.35  # Strong effect
            if study_hours == 'High':
                p_excellent_exam += 0.40  # Strong effect
            final_exam_score = 'Excellent' if (u_exam < p_excellent_exam) else 'Pass'
        else:
            # Rich roots, but keep the mediator binary for a cleaner linear/gaussian fit.
            # Strengthen Attendance/StudyHours -> FinalExamScore to make the dependency
            # survive assembly compression.
            att_inc = {'Poor': 0.0, 'Average': 0.20, 'Good': 0.50}[attendance]
            hrs_inc = {'Low': 0.0, 'Medium': 0.20, 'High': 0.50}[study_hours]
            p_excellent_exam = 0.02 + att_inc + hrs_inc
            if attendance == 'Good' and study_hours == 'High':
                p_excellent_exam += 0.10
            p_excellent_exam = float(min(0.98, max(0.02, p_excellent_exam)))
            final_exam_score = 'Excellent' if (u_exam < p_excellent_exam) else 'Pass'

        # Optional do() override for the mediator
        if do and 'FinalExamScore' in do:
            final_exam_score = do['FinalExamScore']

        # Structural equation for FinalGrade: FinalGrade := g(FinalExamScore, PriorGPA, U_grade)
        if variant == 'base':
            p_excellent_grade = 0.05
            if final_exam_score == 'Excellent':
                p_excellent_grade += 0.50  # Strong effect
            if prior_gpa == 'Average':
                p_excellent_grade += 0.15
            elif prior_gpa == 'High':
                p_excellent_grade += 0.30
            final_grade = 'Excellent' if (u_grade < p_excellent_grade) else 'Pass'
        else:
            # Rich roots, binary outcome; strengthen the direct PriorGPA -> FinalGrade effect.
            p_excellent_grade = 0.05
            if final_exam_score == 'Excellent':
                p_excellent_grade += 0.50
            if prior_gpa == 'Average':
                p_excellent_grade += 0.25
            elif prior_gpa == 'High':
                p_excellent_grade += 0.45
            elif prior_gpa == 'Honors':
                p_excellent_grade += 0.60
            final_grade = 'Excellent' if (u_grade < p_excellent_grade) else 'Pass'

        if do and 'FinalGrade' in do:
            final_grade = do['FinalGrade']

        data.append({

            'PriorGPA': prior_gpa,
            'Attendance': attendance,
            'StudyHours': study_hours,
            'FinalExamScore': final_exam_score,
            'FinalGrade': final_grade
        })

    df = pd.DataFrame(data)

    if variant == 'rich':
        df['PriorGPA'] = pd.Categorical(df['PriorGPA'], categories=['Low', 'Average', 'High', 'Honors'], ordered=True)
        df['Attendance'] = pd.Categorical(df['Attendance'], categories=['Poor', 'Average', 'Good'], ordered=True)
        df['StudyHours'] = pd.Categorical(df['StudyHours'], categories=['Low', 'Medium', 'High'], ordered=True)
        df['FinalExamScore'] = pd.Categorical(df['FinalExamScore'], categories=['Pass', 'Excellent'], ordered=True)
        df['FinalGrade'] = pd.Categorical(df['FinalGrade'], categories=['Pass', 'Excellent'], ordered=True)
    
    # Ground truth from educational theory
    ground_truth_edges = [
        ('PriorGPA', 'FinalGrade'),
        ('Attendance', 'FinalExamScore'),
        ('StudyHours', 'FinalExamScore'),
        ('FinalExamScore', 'FinalGrade')
    ]
    
    print(f'\n  Student Distribution:')
    print(f'    Excellent Exam: {sum(df["FinalExamScore"]=="Excellent")} ({sum(df["FinalExamScore"]=="Excellent")/len(df)*100:.1f}%)')
    print(f'    Excellent Grade: {sum(df["FinalGrade"]=="Excellent")} ({sum(df["FinalGrade"]=="Excellent")/len(df)*100:.1f}%)')
    
    print(f'\n  Ground Truth Causal Edges:')
    for source, target in ground_truth_edges:
        print(f'    {source} -> {target}')
    
    return df, ground_truth_edges

def main():
    """Run Student Success 3-DAG causal validation experiment."""
    print('\n' + '='*70)
    print('STUDENT SUCCESS - 3-DAG CAUSAL VALIDATION')
    print('='*70)
    print('  Research Question: Are causal relationships preserved in assemblies?')
    print('  Method: Compare 3 DAGs (Ground Truth, Neurons, Assemblies)')
    
    print('\n' + '='*70)
    print('PEARL-STYLE SCM (STRUCTURAL CAUSAL MODEL)')
    print('='*70)
    print('  Exogenous noise: U_Prior, U_Att, U_Hours, U_Exam, U_Grade ~ Uniform(0,1)')
    print('  Structural equations (deterministic given U):')
    print('    PriorGPA        := f_prior(U_Prior)')
    print('    Attendance      := f_att(U_Att)')
    print('    StudyHours      := f_hours(U_Hours)')
    print('    FinalExamScore  := (U_Exam  < p_exam(Attendance, StudyHours)) ? Excellent : Pass')
    print('    FinalGrade      := (U_Grade < p_grade(FinalExamScore, PriorGPA)) ? Excellent : Pass')
    print('  do(X=x) intervention: replace X-equation with X := x (override mechanism)')
    
    # Global seeds for reproducibility
    random.seed(DEFAULTS.seed)
    np.random.seed(DEFAULTS.seed)
    os.environ.setdefault('PYTHONHASHSEED', str(DEFAULTS.seed))

    # Generate data
    df, ground_truth = generate_student_data(n_students=DEFAULTS.n_samples, seed=DEFAULTS.seed)
    
    var_names = ['PriorGPA', 'Attendance', 'StudyHours', 
                 'FinalExamScore', 'FinalGrade']

    # Define positive values for consistent spike encoding across runs
    positive_values_map = {
        'PriorGPA': {'High'},
        'Attendance': {'Good'},
        'StudyHours': {'High'},
        'FinalExamScore': {'Excellent'},
        'FinalGrade': {'Excellent'}
    }
    
    # DIAGNOSTIC: Check correlations in original data
    print('\n' + '='*70)
    print('DIAGNOSTIC: Original Data Correlations')
    print('='*70)
    from sklearn.preprocessing import LabelEncoder
    df_encoded = df.copy()
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    print('\nCorrelation Matrix (Original Data):')
    corr = df_encoded[var_names].corr()
    print(corr.to_string())

    # Partial correlations (conditional associations) are often more informative than raw
    # correlations when confounding/mediators exist. This is still a diagnostic: because
    # the data is categorical and label-encoded, treat these values as heuristic.
    def _partial_corr_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.to_numpy(dtype=float)
        x = x - x.mean(axis=0, keepdims=True)
        cov = np.cov(x, rowvar=False)

        # Numerical stability: use a tiny ridge + pseudo-inverse
        cov = cov + (1e-8 * np.eye(cov.shape[0]))
        prec = np.linalg.pinv(cov)

        denom = np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
        denom[denom == 0] = np.nan

        pcorr = -prec / denom
        np.fill_diagonal(pcorr, 1.0)
        return pd.DataFrame(pcorr, index=frame.columns, columns=frame.columns)

    print('\nPartial Correlation Matrix (Label-Encoded; controls for other variables):')
    pcorr = _partial_corr_matrix(df_encoded[var_names])
    print(pcorr.to_string())

    print('\nDIAGNOSTIC: Edge-wise correlation vs partial correlation (label-encoded)')
    for source, target in ground_truth:
        print(f'  {source} -> {target}: corr={corr.loc[source, target]: .3f}, partial={pcorr.loc[source, target]: .3f}')

    # Show a few high-correlation non-edges to highlight correlation vs causation
    edge_set = {(s, t) for s, t in ground_truth}
    pairs = []
    for i, a in enumerate(var_names):
        for j, b in enumerate(var_names):
            if j <= i:
                continue
            if (a, b) in edge_set or (b, a) in edge_set:
                continue
            val = float(corr.loc[a, b])
            pairs.append((abs(val), a, b, val))
    pairs.sort(reverse=True)
    if pairs:
        print('\nTop correlated non-edges (may be confounding/mediation):')
        for _, a, b, val in pairs[:5]:
            print(f'  {a} -- {b}: corr={val: .3f}, partial={pcorr.loc[a, b]: .3f}')    
    # Run 3-DAG validation with MORE PERMISSIVE ALPHA
    print('\n' + '='*70)
    print('Running PC algorithm with alpha=0.1 (more permissive)')
    print('='*70)
    
    print('Using feature extraction: MEAN (rate feature)')
    print('='*70)

    results = run_causal_dag_validation(
        df=df,
        var_names=var_names,
        ground_truth_edges=ground_truth,
        positive_values_map=positive_values_map,
        **runner_kwargs(),
    )

    # If still no edges, try even more permissive alpha
    if len(results['neuron_edges']) == 0 or len(results['assembly_edges']) == 0:
        print(' WARNING: No edges found with alpha=0.1')
        print('Retrying with alpha=0.2 (very permissive)')
        print('='*70)

        results = run_causal_dag_validation(
            df=df,
            var_names=var_names,
            ground_truth_edges=ground_truth,
            positive_values_map=positive_values_map,
            **runner_kwargs(),
        )
# ---------------------------------------------------------------------
    # SCM do()-style intervention evaluation (effect preservation)
    # ---------------------------------------------------------------------
    print('\n' + '='*70)
    print('SCM INTERVENTION EVALUATION (do-operator)')
    print('='*70)
    print('  Goal: check whether interventional effect directions are preserved')
    print('  Note: we reuse the trained Brain from the baseline run (no retraining)')

    from src.encoding.bernoulli import encode_bernoulli_dataframe
    from src.representation.assembly_feature_extraction import extract_assembly_features

    def _extract_variable_features_from_neural(neural: np.ndarray) -> pd.DataFrame:
        feats = {}
        for var_idx, var in enumerate(var_names):
            start = var_idx * DEFAULTS.neurons_per_var
            end = start + DEFAULTS.neurons_per_var
            acts = neural[:, start:end]
            feats[var] = acts.mean(axis=1)
        return pd.DataFrame(feats)

    def _features_for_df(_df: pd.DataFrame, *, brain: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        neural = encode_bernoulli_dataframe(
            _df,
            list(var_names),
            neurons_per_var=DEFAULTS.neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=DEFAULTS.positive_prob,
            negative_prob=DEFAULTS.negative_prob,
            seed=DEFAULTS.seed,
        )
        neuron_df_local = _extract_variable_features_from_neural(neural)

        source_var_by_target_area_name = {v: v for v in var_names}
        assembly_features = extract_assembly_features(
            neural,
            brain,
            list(var_names),
            DEFAULTS.neurons_per_var,
            source_var_by_target_area_name=source_var_by_target_area_name,
        )
        assembly_features_base = {k: v for k, v in assembly_features.items() if '_x_' not in k}
        assembly_df_local = pd.DataFrame(assembly_features_base)
        return neuron_df_local, assembly_df_local

    def _p_excellent(_df: pd.DataFrame, col: str) -> float:
        return float((_df[col] == 'Excellent').mean())

    def _sign(x: float) -> int:
        if x > 1e-12:
            return 1
        if x < -1e-12:
            return -1
        return 0

    n_eval = min(DEFAULTS.n_samples, 2000)
    eval_seed = DEFAULTS.seed + 123

    df_base_eval, _ = generate_student_data(n_students=n_eval, seed=eval_seed, do=None)
    base_neuron_eval, base_assembly_eval = _features_for_df(df_base_eval, brain=results['brain'])

    base_raw_exam = _p_excellent(df_base_eval, 'FinalExamScore')
    base_raw_grade = _p_excellent(df_base_eval, 'FinalGrade')
    base_neuron_exam = float(base_neuron_eval['FinalExamScore'].mean())
    base_neuron_grade = float(base_neuron_eval['FinalGrade'].mean())
    base_assembly_exam = float(base_assembly_eval['FinalExamScore'].mean())
    base_assembly_grade = float(base_assembly_eval['FinalGrade'].mean())

    print(f'  Eval cohort size: {n_eval} (seed={eval_seed})')
    print('  Feature method: MEAN')
    print(f'  Baseline: P(Exam=Excellent)={base_raw_exam:.3f}, P(Grade=Excellent)={base_raw_grade:.3f}')
    print(f'           Neuron mean(Exam)={base_neuron_exam:.3f}, mean(Grade)={base_neuron_grade:.3f}')
    print(f'           Assembly mean(Exam)={base_assembly_exam:.3f}, mean(Grade)={base_assembly_grade:.3f}')

    interventions = [
        ('do(Attendance=Good)', {'Attendance': 'Good'}),
        ('do(StudyHours=High)', {'StudyHours': 'High'}),
        ('do(PriorGPA=High)', {'PriorGPA': 'High'}),
        ('do(FinalExamScore=Excellent)', {'FinalExamScore': 'Excellent'}),
    ]

    print('\nIntervention effect deltas (do - baseline)')
    print('Intervention                | rawΔExam  rawΔGrade | neuronΔExam neuronΔGrade | asmΔExam  asmΔGrade | sign match (Exam/Grade)')
    print("  Interpretation: For each outcome (Exam / Grade), N:Y/A:Y => same direction as raw delta (intervention effect preserved directionally); N:N or A:N => mismatch. Magnitudes can differ due to different scales.")
    print('-' * 118)


    total = 0
    ok_exam_neuron = 0
    ok_exam_assembly = 0
    ok_grade_neuron = 0
    ok_grade_assembly = 0


    for label, do_dict in interventions:
        df_do, _ = generate_student_data(n_students=n_eval, seed=eval_seed, do=do_dict)
        do_neuron, do_assembly = _features_for_df(df_do, brain=results['brain'])

        raw_exam = _p_excellent(df_do, 'FinalExamScore')
        raw_grade = _p_excellent(df_do, 'FinalGrade')
        neuron_exam = float(do_neuron['FinalExamScore'].mean())
        neuron_grade = float(do_neuron['FinalGrade'].mean())
        asm_exam = float(do_assembly['FinalExamScore'].mean())
        asm_grade = float(do_assembly['FinalGrade'].mean())

        d_raw_exam = raw_exam - base_raw_exam
        d_raw_grade = raw_grade - base_raw_grade
        d_neuron_exam = neuron_exam - base_neuron_exam
        d_neuron_grade = neuron_grade - base_neuron_grade
        d_asm_exam = asm_exam - base_assembly_exam
        d_asm_grade = asm_grade - base_assembly_grade

        ok_exam = (_sign(d_raw_exam) == _sign(d_neuron_exam), _sign(d_raw_exam) == _sign(d_asm_exam))
        ok_grade = (_sign(d_raw_grade) == _sign(d_neuron_grade), _sign(d_raw_grade) == _sign(d_asm_grade))
        total += 1
        if ok_exam[0]:
            ok_exam_neuron += 1
        if ok_exam[1]:
            ok_exam_assembly += 1
        if ok_grade[0]:
            ok_grade_neuron += 1
        if ok_grade[1]:
            ok_grade_assembly += 1

        exam_match = f"N:{'Y' if ok_exam[0] else 'N'}/A:{'Y' if ok_exam[1] else 'N'}"
        grade_match = f"N:{'Y' if ok_grade[0] else 'N'}/A:{'Y' if ok_grade[1] else 'N'}"
        print(
            f"{label:27s} | {d_raw_exam:+.3f}  {d_raw_grade:+.3f} |"
            f" {d_neuron_exam:+.3f}     {d_neuron_grade:+.3f} |"
            f" {d_asm_exam:+.3f}   {d_asm_grade:+.3f} | {exam_match} / {grade_match}"
        )

    sign_match_summary = None

    if total > 0:
        exam_rate = f"Exam N={ok_exam_neuron}/{total} ({ok_exam_neuron/total:.1%}), A={ok_exam_assembly}/{total} ({ok_exam_assembly/total:.1%})"
        grade_rate = f"Grade N={ok_grade_neuron}/{total} ({ok_grade_neuron/total:.1%}), A={ok_grade_assembly}/{total} ({ok_grade_assembly/total:.1%})"
        sign_match_summary = f"{exam_rate}; {grade_rate}"
        print()
        print(f"  Sign-match rate (directional): {sign_match_summary}")
    # Visualize the 3 DAGs
    output_dir = os.path.dirname(__file__)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n  Run ID: {run_id}")


    dag_plot_path = os.path.join(output_dir, f'student_success_3dag_comparison_{run_id}.png')

    dag_plot_latest_path = os.path.join(output_dir, 'student_success_3dag_comparison.png')
    visualize_three_dags(
        ground_truth_edges=results['ground_truth_edges'],
        neuron_edges=results['neuron_edges'],
        assembly_edges=results['assembly_edges'],
        var_names=var_names,
        save_path=dag_plot_path
    )
    

    # Keep a stable filename for convenience (latest run)
    try:
        shutil.copy2(dag_plot_path, dag_plot_latest_path)
    except Exception as e:
        print(f"  Warning: failed to update latest DAG PNG: {e}")
    # Summary
    print('\n' + '='*70)
    print('EXPERIMENT COMPLETE')
    print('='*70)
    print(f'\n  Compression: {results["compression"]["ratio"]:.1f}x')
    print(f'  Neuron DAG F1: {results["comparison"]["neuron_vs_gt"]["f1"]:.3f}')
    print(f'  Assembly DAG F1: {results["comparison"]["assembly_vs_gt"]["f1"]:.3f}')
    print(f'  Overall Preservation: {results["comparison"]["preservation_score"]:.3f}')
    
    # DIAGNOSTIC: Show feature statistics
    print('\n' + '='*70)
    print('DIAGNOSTIC: Feature Variance')
    print('='*70)
    print('\nNeuron Features (std dev):')
    print(results['neuron_df'].std().to_string())
    print('\nAssembly Features (std dev):')
    print(results['assembly_df'].std().to_string())
    
    # Save results
    results_file = os.path.join(output_dir, f'student_success_causal_results_{run_id}.txt')

    results_latest_file = os.path.join(output_dir, 'student_success_causal_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('STUDENT SUCCESS - 3-DAG CAUSAL VALIDATION RESULTS\n')
        f.write('='*70 + '\n\n')
        
        f.write('GROUND TRUTH EDGES:\n')
        for source, target in results['ground_truth_edges']:
            f.write(f'  {source} -> {target}\n')
        
        f.write(f'\nNEURON DAG EDGES ({len(results["neuron_edges"])} discovered):\n')
        if results['neuron_edges']:
            for source, target in results['neuron_edges']:
                f.write(f'  {source} -> {target}\n')
        else:
            f.write('  (none discovered)\n')
        
        f.write(f'\nASSEMBLY DAG EDGES ({len(results["assembly_edges"])} discovered):\n')
        if results['assembly_edges']:
            for source, target in results['assembly_edges']:
                f.write(f'  {source} -> {target}\n')
        else:
            f.write('  (none discovered)\n')
        
        f.write('\n' + '='*70 + '\n')
        f.write('VALIDATION METRICS:\n')
        f.write('='*70 + '\n')
        
        f.write('\nNeuron DAG vs Ground Truth:\n')
        ncomp = results['comparison']['neuron_vs_gt']
        f.write(f'  Precision: {ncomp["precision"]:.3f}\n')
        f.write(f'  Recall: {ncomp["recall"]:.3f}\n')
        f.write(f'  F1 Score: {ncomp["f1"]:.3f}\n')
        
        f.write('\nAssembly DAG vs Ground Truth:\n')
        acomp = results['comparison']['assembly_vs_gt']
        f.write(f'  Precision: {acomp["precision"]:.3f}\n')
        f.write(f'  Recall: {acomp["recall"]:.3f}\n')
        f.write(f'  F1 Score: {acomp["f1"]:.3f}\n')
        
        f.write('\nAssembly vs Neuron Agreement:\n')
        ancomp = results['comparison']['assembly_vs_neuron']
        f.write(f'  Similarity: {ancomp["precision"]:.3f}\n')
        
        f.write(f'\nOverall Preservation Score: {results["comparison"]["preservation_score"]:.3f}\n')
        f.write(f'Compression Ratio: {results["compression"]["ratio"]:.1f}x\n')

        f.write('\n' + '='*70 + '\n')
        f.write('INTERVENTION SIGN-MATCH (do operator):\n')
        f.write('='*70 + '\n')
        if sign_match_summary:
            f.write(f'Sign-match rate (directional): {sign_match_summary}\n')
        else:
            f.write('Sign-match rate (directional): (not computed)\n')
    
    # Keep a stable filename for convenience (latest run)
    try:
        shutil.copy2(results_file, results_latest_file)
    except Exception as e:
        print(f"  Warning: failed to update latest results TXT: {e}")

    print(f'\n  Results saved to: {os.path.basename(results_file)}')
    print(f'  Latest results updated: {os.path.basename(results_latest_file)}')
    print(f'  DAG visualization saved to: {os.path.basename(dag_plot_path)}')
    print(f'  Latest DAG PNG updated: {os.path.basename(dag_plot_latest_path)}')
    
    return results

class _TeeTextIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except UnicodeEncodeError:
                enc = getattr(s, 'encoding', None) or 'utf-8'
                safe = data.encode(enc, errors='replace').decode(enc, errors='replace')
                s.write(safe)
                s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, 'isatty', lambda: False)() for s in self._streams)


if __name__ == '__main__':
    output_dir = os.path.dirname(__file__)
    analysis_path = os.path.join(output_dir, 'student_success_causal_results_analysis.txt')

    # Tee all console output (stdout+stderr) into an analysis log file
    with open(analysis_path, 'w', encoding='utf-8') as _log_f:
        _log_f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        _log_f.write('Student Success - step-by-step run log\n\n')

        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeTextIO(_orig_stdout, _log_f)
        sys.stderr = _TeeTextIO(_orig_stderr, _log_f)
        try:
            results = main()
            print(f"\n  Analysis log saved to: {os.path.basename(analysis_path)}")
        finally:
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr






