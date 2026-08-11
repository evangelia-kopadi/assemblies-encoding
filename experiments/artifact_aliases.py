"""Artifact naming utilities for canonical and table-alias filenames."""

from __future__ import annotations

import shutil
from pathlib import Path


TABLE_ALIAS_BY_CANONICAL: dict[str, str] = {
    "single_run_pc_metrics.csv": "table_2_single_run_pc_metrics.csv",
    "single_run_pc_means.csv": "table_2_single_run_pc_means.csv",
    "single_run_ges_metrics.csv": "table_2_single_run_ges_metrics.csv",
    "single_run_ges_means.csv": "table_2_single_run_ges_means.csv",
    "encoding_ablation_pc_raw.csv": "table_3_pc_raw.csv",
    "encoding_ablation_pc_summary.csv": "table_3_pc_summary.csv",
    "encoding_ablation_pc_overall.csv": "table_3_pc_overall.csv",
    "encoding_ablation_ges_raw.csv": "table_3_ges_raw.csv",
    "encoding_ablation_ges_summary.csv": "table_3_ges_summary.csv",
    "encoding_ablation_ges_overall.csv": "table_3_ges_overall.csv",
    "practical_success_rate_summary.csv": "table_4_practical_success_rate_summary.csv",
    "intervention_multiseed_summary.csv": "table_5_interventional_summary.csv",
    "intervention_multiseed_dodeltas.csv": "table_5_interventional_dodeltas.csv",
    "intervention_multiseed_report.md": "table_5_interventional_report.md",
    "interventional_multiseed_summary.csv": "table_5_interventional_summary.csv",  # legacy name
    "interventional_multiseed_dodeltas.csv": "table_5_interventional_dodeltas.csv",  # legacy name
    "interventional_multiseed_report.md": "table_5_interventional_report.md",  # legacy name
    "interventional_summary.csv": "table_5_interventional_summary.csv",  # legacy name
    "interventional_dodeltas.csv": "table_5_interventional_dodeltas.csv",  # legacy name
    "interventional_report.md": "table_5_interventional_report.md",  # legacy name
}


def alias_for(filename: str) -> str | None:
    """Return table alias filename for a canonical artifact, if defined."""
    return TABLE_ALIAS_BY_CANONICAL.get(filename)


def write_table_alias_copy(path: Path) -> Path | None:
    """Create/update table-alias copy next to a canonical artifact."""
    alias_name = alias_for(path.name)
    if alias_name is None:
        return None
    alias_path = path.with_name(alias_name)
    shutil.copy2(path, alias_path)
    return alias_path
