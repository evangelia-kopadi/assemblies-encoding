import json
from pathlib import Path

import experiments.experiment_defaults as defaults


def _reset_output_state(monkeypatch, tmp_path):
    config_file = tmp_path / "experiments_configuration.json"
    config_file.write_text(json.dumps({"example": True}), encoding="utf-8")
    monkeypatch.setattr(defaults, "_out", {"results_root": str(tmp_path / "runs")})
    monkeypatch.setattr(defaults, "_CFG_FILE", str(config_file))
    monkeypatch.setattr(defaults, "_CONFIG_COPIED", False)
    defaults._RUN_METADATA_RECORDED_DIRS.clear()
    monkeypatch.setattr(
        defaults,
        "_get_git_metadata",
        lambda: {
            "commit_sha": "abc123",
            "branch": "test-branch",
            "is_dirty_tracked": False,
        },
    )
    monkeypatch.setattr(
        defaults,
        "_detect_dependency_mode",
        lambda: {"mode": "requirements-lock.txt", "source": "test"},
    )


def test_get_output_filepath_writes_config_snapshot_and_run_metadata(tmp_path, monkeypatch):
    _reset_output_state(monkeypatch, tmp_path)
    monkeypatch.setattr(defaults.sys, "argv", ["experiments/example.py", "--seed", "42"])

    artifact_path = Path(defaults.get_output_filepath("artifact.csv"))
    run_dir = artifact_path.parent

    assert artifact_path.name == "artifact.csv"
    assert (run_dir / "experiments_configuration.json").exists()

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_directory"] == str(run_dir)
    assert metadata["config_snapshot"] == "experiments_configuration.json"
    assert metadata["git"] == {
        "commit_sha": "abc123",
        "branch": "test-branch",
        "is_dirty_tracked": False,
    }
    assert metadata["dependency_mode"] == {"mode": "requirements-lock.txt", "source": "test"}
    assert metadata["python"]["executable"] == defaults.sys.executable
    assert metadata["commands"][0]["argv"] == ["experiments/example.py", "--seed", "42"]

    defaults.get_output_filepath("second.csv")
    metadata_after_second_path = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert len(metadata_after_second_path["commands"]) == 1


def test_run_metadata_appends_command_for_new_process_invocation(tmp_path, monkeypatch):
    _reset_output_state(monkeypatch, tmp_path)
    monkeypatch.setattr(defaults.sys, "argv", ["first.py"])
    run_dir = Path(defaults.get_output_filepath("first.csv")).parent

    defaults._RUN_METADATA_RECORDED_DIRS.clear()
    monkeypatch.setattr(defaults.sys, "argv", ["second.py", "--method", "ges"])
    defaults.get_output_filepath("second.csv")

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert [command["argv"] for command in metadata["commands"]] == [
        ["first.py"],
        ["second.py", "--method", "ges"],
    ]
