"""Evaluate the preregistered criteria (preregistration.md) on the sweep results.

  python experiments/readout-verification/analyse.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARGIN, N_BOOT = 0.05, 10_000


def boot_ci(cell_values: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(cell_values)
    means = cell_values[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return float(cell_values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cell_means(df, arm, col):
    sub = df[df.Arm == arm]
    return sub.groupby(["Dataset", "Seed"])[col].mean()


def separation(setting: str) -> float:
    pp = float(setting.split("pos")[1].split("_")[0])
    pn = float(setting.split("neg")[1])
    return pp - pn


def analyse(method: str) -> list[str]:
    path = HERE / "results" / f"readout_sweep_{method}.csv"
    if not path.exists():
        return [f"## {method.upper()}: no results file ({path.name})"]
    df = pd.read_csv(path)
    lines = [f"## {method.upper()}  ({len(df)} runs, commits {sorted(df.Commit.astype(str).unique())})", ""]

    # R0 replication
    stored = pd.read_csv(ROOT / "results" / f"table_3_{method}_overall.csv").set_index("Config")
    a0 = df[df.Arm == "A0"].groupby("Setting").Assembly_F1.mean()
    lines.append("R0 replication (A0 vs stored mean assembly F1):")
    flag = False
    for s, v in a0.items():
        d = v - stored.loc[s, "Assembly_F1_mean"]
        flag |= abs(d) > MARGIN
        lines.append(f"  {s}: {v:.3f} vs {stored.loc[s, 'Assembly_F1_mean']:.3f} (diff {d:+.3f})")
    lines += [f"  R0: {'FLAGGED (difference > 0.05)' if flag else 'within 0.05'}", ""]

    def noninf(arm, label):
        m, lo, hi = boot_ci(cell_means(df, arm, "Gap").to_numpy())
        ok = lo >= -MARGIN
        lines.append(f"{label}: mean gap {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'MET' if ok else 'NOT MET'}")
        return ok

    h1 = noninf("A1", "H1 (A1 gap non-inferior, margin 0.05)")
    diff = (cell_means(df, "A1", "Assembly_F1") - cell_means(df, "A0", "Assembly_F1")).to_numpy()
    m, lo, hi = boot_ci(diff)
    h2 = lo > 0
    lines.append(f"H2 (A1 - A0 assembly F1 > 0): {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'MET' if h2 else 'NOT MET'}")
    h3 = noninf("FC", "H3 (FC gap non-inferior, margin 0.05)")
    lines.append("")
    noninf("A2", "Secondary: A2 gap (H1 criterion)")
    dk = cell_means(df, "DK", "Assembly_F1")
    m, lo, hi = boot_ci((cell_means(df, "A1", "Assembly_F1") - dk).to_numpy())
    lines.append(f"Secondary: A1 - DK assembly F1: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for arm in ("A0", "A1"):
        g = df[df.Arm == arm].groupby("Setting").Assembly_F1.mean()
        rho = spearmanr([separation(s) for s in g.index], g.values).statistic
        lines.append(f"Secondary: Spearman(p+ - p-, setting mean F1) under {arm}: {rho:+.2f}")
    lines += ["", "Means per setting and arm (assembly F1 / gap / unseen share):"]
    tab = df.groupby(["Setting", "Arm"])[["Neuron_F1", "Assembly_F1", "Gap", "Unseen_share"]].mean().round(3)
    lines += ["  " + l for l in tab.to_string().splitlines()]
    lines += ["", "Means per dataset and arm (gap):"]
    tab = df.pivot_table(index="Dataset", columns="Arm", values="Gap", aggfunc="mean").round(3)
    lines += ["  " + l for l in tab.to_string().splitlines()]
    lines += ["", f"Verdict inputs: H1={'met' if h1 else 'not met'}, H2={'met' if h2 else 'not met'}, "
                  f"H3={'met' if h3 else 'not met'}", ""]
    return lines


if __name__ == "__main__":
    report = ["# Readout verification: results", ""]
    for method in ("pc", "ges"):
        report += analyse(method)
    text = "\n".join(report)
    (HERE / "results" / "analysis.md").write_text(text, encoding="utf-8")
    print(text)
