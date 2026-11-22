"""Readout verification sweep (see preregistration.md in this folder).

Re-runs the Chapter 5 / SUM 2026 encoding sensitivity sweep with the assembly
readout's handling of unseen counts varied, and with a fixed-count Bernoulli code.
The library in src/ is not modified: the variants are swapped in for the duration
of each run by replacing the names that src.runner imported.

Arms (per Bernoulli setting):
  A0  original readout: an unseen count falls back to the raw rate c/N
  A1  nearest seen count: an unseen count uses the stimulus of the nearest count
      formed for that variable (ties: the smaller count)
  A2  new stimulus, no learning: an unseen count gets weights drawn as the Brain
      draws them for a new stimulus (Binomial(count, p) per support neuron),
      without changing the network; the readout is the same top-k mean
  FC  fixed-count Bernoulli code (random active positions each row, exactly
      round(N p+) or round(N p-) of them) with the original readout
plus deterministic-k (step 10) with the original readout as the reference.

  python -m experiments.readout-verification.readout_sweep --method pc
(run from the repository root; the folder name has a hyphen, so use the path form:)
  python experiments/readout-verification/readout_sweep.py --method pc
"""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.runner as runner  # noqa: E402
from src.encoding import bernoulli as bern  # noqa: E402
from src.representation import assembly_feature_extraction as afe  # noqa: E402
from experiments.experiment_defaults import DEFAULTS  # noqa: E402
from experiments.generate_table3_multiseed_run_artifacts import build_datasets  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SETTINGS = [(0.20, 0.10), (0.30, 0.10), (0.40, 0.10), (0.30, 0.05), (0.30, 0.15)]
SEEDS = [42, 73, 101, 131, 151, 181, 211, 241, 271, 301]
DATASETS = ["Alzheimers", "Stroke", "Credit", "Student", "Vaccine"]

_ORIGINAL_EXTRACT = afe.extract_assembly_features
_ORIGINAL_ENCODE = bern.encode_bernoulli_dataframe
_STATS: dict = {}


# --------------------------------------------------------------------------- readouts
def _seen_counts(brain, var):
    out = []
    for name in brain.stimulus_size_by_name:
        v, _, c = name.rpartition("_n")
        if v == var and c.isdigit():
            out.append(int(c))
    return np.array(sorted(out), dtype=int)


def make_extract(mode: str, seed: int):
    """Assembly readout with a given treatment of unseen counts (A0, A1, A2)."""

    def extract(neural_data, brain, var_names, neurons_per_var=1666, *,
                source_var_by_target_area_name=None, top_k=None):
        rng = np.random.default_rng(seed + 7)
        drawn: dict = {}
        feats = {}
        n_rows, n_unseen = neural_data.shape[0], 0
        name_to_idx = {v: i for i, v in enumerate(var_names)}
        for area_name in var_names:
            src = (source_var_by_target_area_name or {}).get(area_name, area_name)
            start = name_to_idx[src] * neurons_per_var
            area = brain.area_by_name[area_name]
            k = int(top_k) if top_k is not None else int(getattr(area, "k", 0) or 0)
            seen = _seen_counts(brain, src)
            counts = neural_data[:, start:start + neurons_per_var].sum(axis=1).astype(int)
            vals = np.zeros(n_rows)
            for i, c in enumerate(counts):
                if c <= 0:
                    continue
                name = f"{src}_n{c}"
                conn = brain.connectomes_by_stimulus.get(name, {}).get(area_name)
                if conn is None or getattr(conn, "size", 0) == 0:
                    n_unseen += 1
                    if mode == "A0":
                        vals[i] = c / neurons_per_var
                        continue
                    if mode == "A1":
                        if len(seen) == 0:
                            vals[i] = c / neurons_per_var
                            continue
                        near = int(seen[np.argmin(np.abs(seen - c))])
                        conn = brain.connectomes_by_stimulus[f"{src}_n{near}"][area_name]
                    elif mode == "A2":
                        key = (src, c)
                        if key not in drawn:
                            size = area.n if area.explicit else area.w
                            drawn[key] = rng.binomial(c, brain.p, size=size).astype(float)
                        conn = drawn[key]
                w = np.asarray(conn, dtype=float)
                if k > 0 and w.size > k:
                    vals[i] = float(w[np.argpartition(w, -k)[-k:]].mean())
                else:
                    vals[i] = float(w.mean())
            feats[area_name] = vals
        _STATS["unseen_share"] = n_unseen / max(1, n_rows * len(var_names))
        return feats

    return extract


# --------------------------------------------------------------------------- encoder
def fixed_count_encode(df, variables, neurons_per_var=1666, positive_prob=0.30,
                       negative_prob=0.10, positive_values_map=None, seed=None):
    """Bernoulli-like code with the count fixed per class: each row activates a
    fresh random set of exactly round(N p+) (positive) or round(N p-) neurons."""
    n = len(df)
    out = np.zeros((n, len(variables) * neurons_per_var), dtype=int)
    k_pos, k_neg = round(neurons_per_var * positive_prob), round(neurons_per_var * negative_prob)
    for j, var in enumerate(variables):
        rng = np.random.default_rng(None if seed is None else seed + j)
        values = df[var].values
        pos = (positive_values_map or {}).get(var)
        if pos is None:
            uniq = sorted(set(values), key=str)
            pos = {uniq[1]} if len(uniq) == 2 else bern.DEFAULT_POSITIVE_VALUES
        for i, v in enumerate(values):
            kk = k_pos if v in pos else k_neg
            out[i, j * neurons_per_var + rng.choice(neurons_per_var, kk, replace=False)] = 1
    return out


@contextlib.contextmanager
def variant(readout: str, seed: int, fixed_count: bool):
    runner.extract_assembly_features = make_extract(readout, seed)
    if fixed_count:
        runner.encode_bernoulli_dataframe = fixed_count_encode
    try:
        yield
    finally:
        runner.extract_assembly_features = _ORIGINAL_EXTRACT
        runner.encode_bernoulli_dataframe = _ORIGINAL_ENCODE


# --------------------------------------------------------------------------- sweep
def arms():
    out = []
    for pp, pn in SETTINGS:
        tag = f"Bernoulli_pos{pp:.2f}_neg{pn:.2f}"
        for r in ("A0", "A1", "A2"):
            out.append(dict(arm=r, setting=tag, readout=r, fixed_count=False,
                            enc=dict(deterministic_k_encoding=False, positive_prob=pp, negative_prob=pn)))
        out.append(dict(arm="FC", setting=tag, readout="A0", fixed_count=True,
                        enc=dict(deterministic_k_encoding=False, positive_prob=pp, negative_prob=pn)))
    out.append(dict(arm="DK", setting="Deterministic_kstep10", readout="A0", fixed_count=False,
                    enc=dict(deterministic_k_encoding=True, deterministic_k_step=10,
                             deterministic_k_readout_mode="pool_mean", stimulus_k=DEFAULTS.assembly_k)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pc", "ges"], required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    datasets = build_datasets(800)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    base = dict(neurons_per_var=DEFAULTS.neurons_per_var, assembly_k=DEFAULTS.assembly_k,
                n_train=120, n_presentations=3, beta=DEFAULTS.beta, alpha_pc=0.05,
                strict_neuron_causal_discovery=False, strict_assembly_causal_discovery=False)
    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"readout_sweep_{a.method}{a.tag}.csv"
    rows = []
    for ds in a.datasets.split(","):
        d = datasets[ds]
        for seed in seeds:
            with contextlib.redirect_stdout(io.StringIO()):
                df, gt = d["gen"](seed)
            for arm in arms():
                t = time.time()
                _STATS.clear()
                with variant(arm["readout"], seed, arm["fixed_count"]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    res = runner.run_causal_dag_validation(
                        method=a.method, df=df, var_names=d["var_names"], ground_truth_edges=gt,
                        positive_values_map=d["positive_values_map"], seed=seed, **base, **arm["enc"])
                comp = res.get("comparison", {})
                nf = float((comp.get("neuron_vs_gt") or {}).get("f1", np.nan))
                af = float((comp.get("assembly_vs_gt") or {}).get("f1", np.nan))
                rows.append(dict(Dataset=ds, Method=a.method, Seed=seed, Setting=arm["setting"],
                                 Arm=arm["arm"], Neuron_F1=nf, Assembly_F1=af, Gap=af - nf,
                                 Unseen_share=_STATS.get("unseen_share", np.nan),
                                 Seconds=round(time.time() - t, 1), Commit=commit))
                print(f"{ds} seed={seed} {arm['setting']} {arm['arm']}: neuron={nf:.3f} "
                      f"assembly={af:.3f} unseen={rows[-1]['Unseen_share']:.2f} ({rows[-1]['Seconds']}s)",
                      flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
