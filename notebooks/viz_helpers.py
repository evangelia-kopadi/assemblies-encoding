from __future__ import annotations

import math as _math

import numpy as np
import pandas as pd
from matplotlib.patches import Circle as _Circle, FancyArrowPatch as _FAP

from src.discovery.ges import run_ges_algorithm
from src.discovery.pc import run_pc_algorithm
from src.encoding.bernoulli import encode_bernoulli_dataframe
from src.encoding.deterministic_k import build_deterministic_k_map, encode_deterministic_k_dataframe
from src.representation.assembly_feature_extraction import extract_assembly_features
from src.representation.assembly_formation import form_assemblies
from src.representation.brain import Brain


def encode(
    df,
    *,
    deterministic_k: bool,
    var_names,
    assembly_k: int,
    neurons_per_var: int,
    seed: int,
    positive_values_map,
    positive_prob: float = 0.30,
    negative_prob: float = 0.10,
    k_step: int = 10,
):
    if deterministic_k:
        k_map = build_deterministic_k_map(df, var_names, base_k=assembly_k, k_step=k_step)
        neural, stim_idx = encode_deterministic_k_dataframe(
            df,
            var_names,
            neurons_per_var=neurons_per_var,
            stimulus_k_map=k_map,
            seed=seed,
        )
        return neural, stim_idx
    neural = encode_bernoulli_dataframe(
        df,
        var_names,
        neurons_per_var=neurons_per_var,
        positive_values_map=positive_values_map,
        positive_prob=positive_prob,
        negative_prob=negative_prob,
        seed=seed,
    )
    return neural, None


def neuron_readout(
    neural,
    *,
    var_names,
    neurons_per_var: int,
    positive_values_map,
    stim_idx=None,
    deterministic_k: bool = False,
):
    features = {}
    for idx, var in enumerate(var_names):
        block = neural[:, idx * neurons_per_var : (idx + 1) * neurons_per_var]
        if deterministic_k and stim_idx and var in stim_idx:
            pos_vals = [v for v in positive_values_map.get(var, set()) if v in stim_idx[var]]
            if pos_vals:
                local = np.concatenate([stim_idx[var][v] for v in pos_vals])
                features[var] = block[:, local].mean(axis=1)
                continue
        features[var] = block.mean(axis=1)
    return pd.DataFrame(features)


def assembly_formation(
    neural,
    *,
    var_names,
    neurons_per_var: int,
    assembly_k: int,
    beta: float,
    seed: int,
    n_train: int,
    n_presentations: int,
):
    brain = Brain(p=beta, save_size=True, save_winners=True, seed=seed)
    for var in var_names:
        brain.add_area(var, n=neurons_per_var, k=assembly_k, beta=beta)
    return form_assemblies(
        brain,
        neural,
        var_names,
        neurons_per_var,
        n_train=n_train,
        n_presentations=n_presentations,
        target_area_by_var_name={v: v for v in var_names},
        shuffle_seed=seed,
    )


def assembly_readout(
    neural,
    brain,
    *,
    var_names,
    neurons_per_var: int,
):
    feats = extract_assembly_features(
        neural,
        brain,
        var_names,
        neurons_per_var,
        source_var_by_target_area_name={v: v for v in var_names},
    )
    return pd.DataFrame({k: v for k, v in feats.items() if "_x_" not in k})


def discover(feature_df, *, var_names, method: str, alpha: float = 0.05, strict: bool = False):
    if method == "pc":
        edges, _ = run_pc_algorithm(feature_df, var_names, alpha=alpha, strict=strict)
    else:
        edges, _ = run_ges_algorithm(feature_df, var_names, strict=strict)
    return edges


def _clip(pos_s, pos_t, r):
    sx, sy = pos_s
    tx, ty = pos_t
    dx, dy = tx - sx, ty - sy
    d = _math.hypot(dx, dy)
    if d == 0:
        return pos_s, pos_t
    ux, uy = dx / d, dy / d
    return (sx + ux * r, sy + uy * r), (tx - ux * (r + 0.02), ty - uy * (r + 0.02))


def draw_dag(
    ax,
    edges,
    title,
    node_color,
    *,
    var_names,
    dag_pos,
    gt_edges=None,
    node_r: float = 0.34,
    label_fs: float = 8.0,
    title_fs: float = 10.0,
    wrap_labels=None,
):
    wrap_labels = wrap_labels or {}

    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=title_fs, pad=8, fontweight="bold")

    gt = set(map(tuple, gt_edges)) if gt_edges else None

    for s, t in edges:
        if s not in dag_pos or t not in dag_pos:
            continue
        if gt is None:
            col = "#37474F"
        elif (s, t) in gt:
            col = "#2E7D32"
        else:
            col = "#C62828"
        start, end = _clip(dag_pos[s], dag_pos[t], node_r)
        patch = _FAP(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.2,
            color=col,
            zorder=1,
        )
        ax.add_patch(patch)

    for var in var_names:
        x, y = dag_pos[var]
        ax.add_patch(
            _Circle(
                (x, y),
                node_r,
                facecolor=node_color,
                edgecolor="#263238",
                linewidth=2.0,
                zorder=2,
            )
        )
        label = wrap_labels.get(var, var)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=label_fs,
            fontweight="semibold",
            zorder=3,
            multialignment="center",
            color="#1a1a1a",
        )

    if gt_edges:
        ax.plot([], [], color="#2E7D32", lw=2.5, label="correct")
        ax.plot([], [], color="#C62828", lw=2.5, label="spurious")
        ax.legend(
            loc="lower center",
            fontsize=7.5,
            framealpha=0.75,
            ncol=2,
            handlelength=1.4,
            borderpad=0.5,
        )
