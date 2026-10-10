# tests/test_data.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.train import build_dataloaders


def test_dataloaders_shapes(base_cfg):
    """Cada batch deve ter shape (B, 1, 28, 28) e labels (B,)."""
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = 32
    cfg["batch_size"] = 8

    train_loader, test_loader = build_dataloaders(cfg)

    xb, yb = next(iter(train_loader))
    assert xb.shape == (8, 1, 28, 28), f"Shape inesperado: {xb.shape}"
    assert yb.shape == (8,)
    assert xb.dtype == torch.float32
    assert yb.dtype == torch.int64


def test_subset_respects_size(base_cfg):
    """subset_size deve limitar o número de amostras de treino."""
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = 100
    cfg["batch_size"] = 16

    train_loader, _ = build_dataloaders(cfg)
    assert len(train_loader.dataset) == 100


def test_subset_is_deterministic(base_cfg):
    """Duas chamadas com mesma seed devem gerar o mesmo subset."""
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = 50
    cfg["seed"] = 123

    train_loader_1, _ = build_dataloaders(cfg)
    train_loader_2, _ = build_dataloaders(cfg)

    idx_1 = list(train_loader_1.dataset.indices)
    idx_2 = list(train_loader_2.dataset.indices)
    assert idx_1 == idx_2, "Subset não é determinístico entre execuções"


def test_normalization_applied(base_cfg):
    """A normalização deve deixar a média próxima de 0 nos dados transformados."""
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = 256
    cfg["batch_size"] = 256

    train_loader, _ = build_dataloaders(cfg)
    xb, _ = next(iter(train_loader))
    # Após normalização com mean/std do MNIST, a média deve estar perto de 0
    assert abs(xb.mean().item()) < 0.5, f"Média inesperada: {xb.mean().item()}"


def test_full_dataset_when_subset_null(base_cfg):
    """subset_size=null deve usar o dataset completo (60k treino)."""
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = None
    cfg["batch_size"] = 128

    train_loader, test_loader = build_dataloaders(cfg)
    assert len(train_loader.dataset) == 60000
    assert len(test_loader.dataset) == 10000


def test_no_workers_for_debug(base_cfg):
    """num_workers=0 é o padrão local — não deve quebrar."""
    cfg = dict(base_cfg)
    cfg["dataset"] = dict(base_cfg["dataset"])
    cfg["dataset"]["subset_size"] = 16
    cfg["num_workers"] = 0

    train_loader, _ = build_dataloaders(cfg)
    xb, yb = next(iter(train_loader))
    assert xb.shape[0] == yb.shape[0]