# tests/test_model.py
import pytest
import torch

from src.model import SimpleCNN, build_model


def test_build_model_returns_simple_cnn(base_cfg):
    model = build_model(base_cfg)
    assert isinstance(model, SimpleCNN)


def test_build_model_unknown_raises(base_cfg):
    cfg = dict(base_cfg)
    cfg["model"] = dict(base_cfg["model"])
    cfg["model"]["name"] = "inexistente"
    with pytest.raises(ValueError, match="Modelo desconhecido"):
        build_model(cfg)


def test_forward_output_shape(base_cfg):
    """Forward em batch (B,1,28,28) deve retornar (B, num_classes)."""
    model = build_model(base_cfg)
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_num_classes_custom():
    model = SimpleCNN(in_channels=1, num_classes=5, dropout=0.0)
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_dropout_zero_is_identity():
    """Com dropout=0, o módulo deve ser Identity (sem aleatoriedade)."""
    model = SimpleCNN(dropout=0.0)
    model.train()
    x = torch.randn(3, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)
    # Sem dropout, duas passadas em train() devem ser idênticas
    assert torch.allclose(out1, out2)


def test_dropout_positive_changes_output():
    """Com dropout>0 em train(), duas passadas devem diferir."""
    torch.manual_seed(0)
    model = SimpleCNN(dropout=0.5)
    model.train()
    x = torch.randn(8, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2), "Dropout não está ativo em train()"


def test_model_parameters_count(base_cfg):
    """Sanidade: o modelo deve ter parâmetros treináveis (>0)."""
    model = build_model(base_cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n > 0
    assert n < 10_000_000, f"Modelo suspeitamente grande: {n} parâmetros"


def test_gradients_flow(base_cfg):
    """Backward deve produzir gradientes não-nulos em todos os pesos."""
    model = build_model(base_cfg)
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Sem gradiente em {name}"
        assert p.grad.abs().sum() > 0, f"Gradiente nulo em {name}"