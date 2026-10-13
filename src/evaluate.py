# src/evaluate.py
"""
Checkpoint evaluation salved by Experiment.

Uso:
    # Avaliar o melhor checkpoint de um experimento específico
    python src/evaluate.py --checkpoint experiments/2026-.../checkpoints/best.pth

    # Avaliar usando GPU
    python src/evaluate.py --checkpoint ... --device cuda

    # Avaliar o experimento mais recente automaticamente
    python src/evaluate.py --latest --experiments-dir experiments/

    # Salvar relatório JSON na pasta do experimento
    python src/evaluate.py --checkpoint ... --save-report

    # Gerar matriz de confusão (requer matplotlib + seaborn)
    python src/evaluate.py --checkpoint ... --confusion-matrix

    Codigo simples para checar um checkpoint:
    
    import torch
    ckpt = torch.load("experiments/.../checkpoints/best.pth", map_location="cpu")
    print("Config:", ckpt["config"])
    print("Git commit:", ckpt["git_commit"])
    print("Dirty:", ckpt["git_dirty"])

"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Importa do pacote src
try:
    from src.train import build_criterion
    from src.model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import build_criterion
    from model import build_model


# --------------------------------------------------------------------------- #
# Checkpoint load
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str, device: torch.device):
    """Carrega checkpoint e reconstroi o modelo usando a config embutida."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    required = ["state_dict", "config"]
    for k in required:
        if k not in ckpt:
            raise ValueError(f"Checkpoint malformado: falta '{k}'")

    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    meta = {
        "config_hash": ckpt.get("config_hash"),
        "git_commit": ckpt.get("git_commit"),
        "git_dirty": ckpt.get("git_dirty"),
        "saved_at": ckpt.get("saved_at"),
        "epoch": ckpt.get("epoch"),
        "val_acc": ckpt.get("val_acc"),
        "is_best": ckpt.get("is_best", False),
    }
    return model, cfg, meta


# --------------------------------------------------------------------------- #
# Test data
# --------------------------------------------------------------------------- #
#def build_test_loader(cfg: dict, batch_size: int = 256, num_workers: int = 0):
#    """Constroi o test loader a partir da config embutida no checkpoint."""
#    ds_cfg = cfg["dataset"]
#    transform = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(ds_cfg["normalize_mean"], ds_cfg["normalize_std"]),
#    ])
#    test_ds = datasets.MNIST(
#        cfg["data_path"], train=False, download=True, transform=transform
#    )
#    # Se o treino usou subset, avaliamos no test set completo mesmo assim.
#    # (Avaliar num subset do test seria menos informativo.)
#    return DataLoader(
#        test_ds, batch_size=batch_size, shuffle=False,
#        num_workers=num_workers, pin_memory=False,
#    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_full(model, loader, device, num_classes: int = 10):
    """Retorna metricas: loss, acc, per-class acc, confusion matrix, top-k."""
    criterion = build_criterion()
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total = 0

    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        total += data.size(0)

        probs = torch.softmax(output, dim=1)
        all_probs.append(probs.cpu())
        all_preds.append(output.argmax(1).cpu())
        all_targets.append(target.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = torch.cat(all_probs).numpy()

    acc = (preds == targets).mean()
    avg_loss = total_loss / total

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class_acc[c] = float((preds[mask] == c).mean())

    # Matriz de confusão (sem sklearn)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    # Top-5 accuracy
    top5 = np.argsort(-probs, axis=1)[:, :5]
    top5_acc = float(np.mean([t in row for t, row in zip(targets, top5)]))

    # Confiança média nas predições corretas vs incorretas
    conf_correct = float(probs[np.arange(len(preds)), preds][preds == targets].mean()) \
        if (preds == targets).any() else 0.0
    conf_wrong = float(probs[np.arange(len(preds)), preds][preds != targets].mean()) \
        if (preds != targets).any() else 0.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "top5_accuracy": top5_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "mean_confidence_correct": conf_correct,
        "mean_confidence_wrong": conf_wrong,
        "n_samples": int(total),
    }


# --------------------------------------------------------------------------- #
# Utilitarios de visualizacao
# --------------------------------------------------------------------------- #
def print_report(meta: dict, metrics: dict, num_classes: int = 10):
    print("\n" + "=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("=" * 60)

    print("\n--- Checkpoint ---")
    print(f"  config_hash : {meta.get('config_hash')}")
    print(f"  git_commit  : {meta.get('git_commit')}")
    print(f"  git_dirty   : {meta.get('git_dirty')}")
    print(f"  saved_at    : {meta.get('saved_at')}")
    print(f"  epoch       : {meta.get('epoch')}")
    print(f"  is_best     : {meta.get('is_best')}")

    print("\n--- Métricas globais ---")
    print(f"  amostras          : {metrics['n_samples']}")
    print(f"  loss              : {metrics['loss']:.4f}")
    print(f"  accuracy          : {metrics['accuracy']:.4f}")
    print(f"  top-5 accuracy    : {metrics['top5_accuracy']:.4f}")
    print(f"  conf. média (acertos) : {metrics['mean_confidence_correct']:.4f}")
    print(f"  conf. média (erros)   : {metrics['mean_confidence_wrong']:.4f}")

    print("\n--- Acurácia por classe ---")
    for c in range(num_classes):
        if c in metrics["per_class_accuracy"]:
            acc = metrics["per_class_accuracy"][c]
            bar = "█" * int(acc * 30)
            print(f"  dígito {c}: {acc:.4f} {bar}")

    print("\n--- Matriz de confusão ---")
    cm = np.array(metrics["confusion_matrix"])
    header = "     " + " ".join(f"{i:5d}" for i in range(num_classes))
    print(header)
    for i in range(num_classes):
        row = " ".join(f"{v:5d}" for v in cm[i])
        print(f"  {i}: {row}")

    # Classes mais confundidas
    print("\n--- Top-5 confusões ---")
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], i, j))
    confusions.sort(reverse=True)
    for count, true_c, pred_c in confusions[:5]:
        print(f"  {count:4d}× verdadeiro={true_c}, predito={pred_c}")

    print("=" * 60 + "\n")


def plot_confusion_matrix(metrics: dict, out_path: Path, num_classes: int = 10):
    """Gera e salva matriz de confusão como imagem."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn não instalados. Pulando plot.")
        return

    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de Confusão — acc={metrics['accuracy']:.4f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[Plot] Matriz salva em: {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def find_latest_checkpoint(experiments_dir: Path) -> Path:
    """Encontra o best.pth do experimento mais recente."""
    runs = sorted(experiments_dir.glob("*/metadata.json"))
    if not runs:
        raise FileNotFoundError(f"Nenhum experimento encontrado em {experiments_dir}")
    latest = runs[-1].parent
    # Prioriza best.pth, senão last.pth, senão o primeiro .pth
    for name in ["best.pth", "last.pth"]:
        p = latest / "checkpoints" / name
        if p.exists():
            return p
    pths = sorted((latest / "checkpoints").glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"Nenhum checkpoint em {latest}")
    return pths[0]


def main():
    parser = argparse.ArgumentParser(description="Avalia um checkpoint salvo pelo Experiment")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Caminho do arquivo .pth")
    parser.add_argument("--latest", action="store_true",
                        help="Usa o best.pth do experimento mais recente")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Diretório base dos experimentos (para --latest)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device de avaliação")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-report", action="store_true",
                        help="Salva eval_report.json na pasta do experimento")
    parser.add_argument("--confusion-matrix", action="store_true",
                        help="Gera imagem da matriz de confusão")
    parser.add_argument("--show-errors", type=int, default=0,
                        help="Mostra N exemplos classificados incorretamente")
    args = parser.parse_args()

    # Resolve o caminho do checkpoint
    if args.latest:
        ckpt_path = find_latest_checkpoint(Path(args.experiments_dir))
        print(f"[CLI] Usando checkpoint mais recente: {ckpt_path}")
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        parser.error("Forneça --checkpoint <path> ou --latest")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA indisponível, caindo para CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[CLI] Device: {device}")

    # Carrega
    model, cfg, meta = load_checkpoint(str(ckpt_path), device)
    print(f"[CLI] Modelo reconstruído a partir da config embutida no checkpoint")

    # Dados
    #loader = build_test_loader(cfg, batch_size=args.batch_size, num_workers=args.num_workers)

    # Avalia
    num_classes = cfg["model"].get("num_classes", 10)
    metrics = evaluate_full(model, loader, device, num_classes=num_classes)

    # Report
    print_report(meta, metrics, num_classes=num_classes)

    # Salva relatório
    if args.save_report:
        report = {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_meta": meta,
            "config_hash": meta.get("config_hash"),
            "metrics": metrics,
        }
        out = ckpt_path.parent.parent / "eval_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"[Report] Salvo em: {out}")


if __name__ == "__main__":
    main()