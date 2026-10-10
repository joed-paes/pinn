# tests/conftest.py
import sys
from pathlib import Path

import pytest
import yaml
import torch

# Garante que `src` seja importável a partir dos testes
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def base_cfg():
    """Config mínima em memória, espelhando default.yaml + local.yaml."""
    return {
        "seed": 42,
        "deterministic": False,
        "data_path": str(ROOT / "data" / "mnist"),
        "experiment_path": str(ROOT / "experiments_test"),
        "checkpoint_path": str(ROOT / "checkpoints_test"),
        "device": "cpu",
        "epochs": 1,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "log_interval": 0,
        "save_best": True,
        "save_last": True,
        "save_every_epoch": False,
        "model": {
            "name": "simple_cnn",
            "in_channels": 1,
            "num_classes": 10,
            "dropout": 0.0,
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "scheduler": {"name": None},
        "dataset": {
            "name": "mnist",
            "normalize_mean": [0.1307],
            "normalize_std": [0.3081],
            "subset_size": 64,  # pequeno para testes rápidos
        },
        "experiment": {"name_prefix": "test"},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def tmp_experiments_dir(tmp_path):
    """Diretório temporário isolado para cada teste que cria Experiment."""
    d = tmp_path / "experiments"
    d.mkdir()
    return d


@pytest.fixture
def tiny_dataloaders(base_cfg):
    """DataLoaders minúsculos para testes rápidos (usa subset se possível)."""
    from src.train import build_dataloaders
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = 32
    return build_dataloaders(cfg)


@pytest.fixture(autouse=True)
def set_test_seed():
    """Semente fixa em todos os testes para reprodutibilidade."""
    torch.manual_seed(0)
    yield