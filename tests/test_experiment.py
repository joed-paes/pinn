# tests/test_experiment.py
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.experiment import Experiment, capture_git_info, config_hash


# --------------------------------------------------------------------------- #
# Testes de funções auxiliares
# --------------------------------------------------------------------------- #
def test_capture_git_info_keys():
    info = capture_git_info()
    for key in ["commit", "branch", "is_dirty", "diff", "status"]:
        assert key in info, f"Chave ausente: {key}"


def test_config_hash_stable():
    """O hash deve ser determinístico, independente da ordem de chaves."""
    cfg1 = {"a": 1, "b": {"c": 2, "d": 3}}
    cfg2 = {"b": {"d": 3, "c": 2}, "a": 1}
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_differs_on_value():
    assert config_hash({"lr": 0.1}) != config_hash({"lr": 0.2})


# --------------------------------------------------------------------------- #
# Testes de criação do Experiment
# --------------------------------------------------------------------------- #
def test_experiment_creates_dirs(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    assert exp.dir.exists()
    assert exp.ckpt_dir.exists()
    assert (exp.dir / "config.resolved.yaml").exists()
    assert (exp.dir / "git_info.yaml").exists()
    assert (exp.dir / "metrics.jsonl").exists()


def test_experiment_saves_resolved_config(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    with open(exp.dir / "config.resolved.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["batch_size"] == base_cfg["batch_size"]
    assert saved["model"]["name"] == base_cfg["model"]["name"]


def test_experiment_name_contains_params(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg, name_prefix="mnist")
    assert "mnist" in exp.name
    # assert "learning_rate=" in exp.name # learning_rate fica dentro da estrutura do optimizer
    assert "batch_size=" in exp.name


def test_experiment_unique_names(base_cfg, tmp_experiments_dir):
    """Duas chamadas não devem colidir (timestamp + hash evitam isso)."""
    import time
    exp1 = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    time.sleep(1.1)  # garante timestamp diferente
    exp2 = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    assert exp1.dir != exp2.dir


# --------------------------------------------------------------------------- #
# Testes de checkpoint
# --------------------------------------------------------------------------- #
def test_save_checkpoint_embeds_metadata(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    state = {"w": torch.randn(3, 3)}
    exp.save_checkpoint(state, name="test.pth", extra_meta={"epoch": 1})

    ckpt_path = exp.ckpt_dir / "test.pth"
    assert ckpt_path.exists()

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "state_dict" in loaded
    assert "config" in loaded
    assert "config_hash" in loaded
    assert "git_commit" in loaded
    assert "git_dirty" in loaded
    assert "saved_at" in loaded
    assert loaded["epoch"] == 1  # extra_meta preservado


def test_checkpoint_config_matches_experiment(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    exp.save_checkpoint({"w": torch.zeros(1)}, name="ckpt.pth")

    loaded = torch.load(exp.ckpt_dir / "ckpt.pth", map_location="cpu", weights_only=False)
    assert loaded["config_hash"] == exp.cfg_hash
    assert loaded["config"]["batch_size"] == base_cfg["batch_size"]


def test_multiple_checkpoints_coexist(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    exp.save_checkpoint({"w": torch.zeros(1)}, name="best.pth")
    exp.save_checkpoint({"w": torch.ones(1)}, name="last.pth")
    exp.save_checkpoint({"w": torch.full((1,), 2.0)}, name="epoch_01.pth")

    files = sorted(p.name for p in exp.ckpt_dir.glob("*.pth"))
    assert files == ["best.pth", "epoch_01.pth", "last.pth"]


# --------------------------------------------------------------------------- #
# Testes de métricas e finalize
# --------------------------------------------------------------------------- #
def test_log_metrics_appends_jsonl(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    exp.log_metrics({"epoch": 1, "loss": 0.5, "acc": 0.9})
    exp.log_metrics({"epoch": 2, "loss": 0.3, "acc": 0.95})

    lines = exp.metrics_path.read_text().strip().split("\n")
    assert len(lines) == 2

    m1 = json.loads(lines[0])
    m2 = json.loads(lines[1])
    assert m1["epoch"] == 1
    assert m2["acc"] == 0.95
    assert "timestamp" in m1 and "timestamp" in m2


def test_finalize_writes_metadata(base_cfg, tmp_experiments_dir):
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    exp.finalize(status="completed", best_acc=0.98, epochs=5)

    meta_path = exp.dir / "metadata.json"
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "completed"
    assert meta["best_acc"] == 0.98
    assert meta["epochs"] == 5
    assert meta["experiment_name"] == exp.name
    assert meta["config_hash"] == exp.cfg_hash
    assert "finished_at" in meta


def test_finalize_overwrites_previous(base_cfg, tmp_experiments_dir):
    """Se finalize for chamado duas vezes, o metadata final prevalece."""
    exp = Experiment(base_dir=str(tmp_experiments_dir), cfg=base_cfg)
    exp.finalize(status="failed", error="interrompido")
    exp.finalize(status="completed", best_acc=0.9)

    meta = json.loads((exp.dir / "metadata.json").read_text())
    assert meta["status"] == "completed"
    assert "error" not in meta or meta.get("error") is None