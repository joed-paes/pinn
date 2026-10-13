# src/train.py
"""
Script principal de treino de PINN.

Uso:
    # Workstation (CPU, subset pequeno)
    python src/train.py --config config/local.yaml

    # GPU-server (CUDA, dataset completo) — dentro do container, via tmux
    python src/train.py --config config/server.yaml

Estrutura gerada por execução:
    <experiment_path>/<timestamp>_<prefix>_lr=..._bs=.../
        ├── config.resolved.yaml
        ├── git_info.yaml
        ├── metrics.jsonl
        ├── train.log
        ├── metadata.json
        └── checkpoints/
            ├── best.pth
            ├── last.pth
            └── epoch_NN.pth
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import deepxde as dde
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Importa do pacote src (funciona tanto rodando como modulo quanto direto)
try:
    from src.experiment import Experiment
    from src.utils import load_config
    from src.model import build_model
    from src.pdes import build_pde
    from src.geometry import SpatialTemporalDomain
except ImportError:
    # Fallback para execucao direta: python src/train.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment import Experiment
    from utils import load_config
    from model import build_model
    from pdes import build_pde
    from geometry import SpatialTemporalDomain


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
#def build_dataloaders(cfg: dict):
#    """Constroi train/test loaders respeitando subset_size e num_workers."""
#    ds_cfg = cfg["dataset"]
#
#   # To be modified when using external data in the case of PINNs
#   train_ds = []
#   test_ds = []

 #   train_loader = DataLoader(
 #       train_ds,
 #       batch_size=cfg["batch_size"],
 #       shuffle=True,
 #       num_workers=cfg["num_workers"],
 #       pin_memory=cfg["pin_memory"],
 #   )
 #   test_loader = DataLoader(
 #       test_ds,
 #       batch_size=cfg["batch_size"],
 #       shuffle=False,
 #       num_workers=cfg["num_workers"],
 #       pin_memory=cfg["pin_memory"],
 #   )
 #   return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Geometry / Boundary
# --------------------------------------------------------------------------- #
def build_geometry(cfg):
    g = cfg["geometry"]
    if g["name"] == "interval_1d_time":
        geom = dde.geometry.Interval(g["x_min"], g["x_max"])
        td = dde.geometry.TimeDomain(g["t_min"], g["t_max"])
        return dde.geometry.GeometryXTime(geom, td)
    if g["name"] == "polygon_2d":
        return dde.geometry.Polygon(g["vertices"])
    raise ValueError(f"Geometria desconhecida: {g['name']}")


def build_conditions(cfg, geom):
    """Retorna lista de condicoes. Ajustar conforme o problema."""
    name = cfg["pde"]["name"]
    conditions = []

    if name == "heat_1d":
        bc = dde.icbc.DirichletBC(geom, lambda x: 0,
                                   lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(geom, lambda x: np.sin(np.pi * x[:, 0:1]),
                          lambda _, on_initial: on_initial)
        conditions = [bc, ic]
    elif name == "poisson_2d":
        bc = dde.icbc.DirichletBC(geom, lambda x: 0,
                                   lambda _, on_boundary: on_boundary)
        conditions = [bc]
    elif name == "burguer":
        bc = []

    return conditions


# --------------------------------------------------------------------------- #
# Criterion / Optimizer / Scheduler
# --------------------------------------------------------------------------- #
def build_criterion(cfg: dict):
    opt_cfg = cfg["criterion"]
    name = opt_cfg["name"].lower()
    if name == "entropy":
        return nn.CrossEntropyLoss()
    if name == "l2_norm":
        return nn.MSELoss()
    raise ValueError(f"Criterio desconhecido: {name}")

def build_optimizer(model, cfg: dict):
    opt_cfg = cfg["optimizer"]
    name = opt_cfg["name"].lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=opt_cfg["learning_rate"],
                          weight_decay=opt_cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=opt_cfg["learning_rate"],
                         momentum=opt_cfg.get("momentum", 0.9),
                         weight_decay=opt_cfg.get("weight_decay", 0.0))
    raise ValueError(f"Otimizador desconhecido: {name}")


def build_scheduler(optimizer, cfg: dict):
    sch_cfg = cfg.get("scheduler") or {}
    name = sch_cfg.get("name")
    if not name:
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=sch_cfg["step_size"], gamma=sch_cfg["gamma"]
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"]
        )
    raise ValueError(f"Scheduler desconhecido: {name}")


# --------------------------------------------------------------------------- #
# Loop de treino / validacao
# --------------------------------------------------------------------------- #
def train_one_epoch(pde_fn, ic_true_fn, bc_true_fn, params, model, sampling, optimizer, criterion, loss_weights ,device, log_interval):
    model.train()

    # If I would use batch within a dataloader, I could add a for loop here to evaluate the model in this batch

    # data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

    # Coordinates
    # Defino o tipo de sampling no arquivo de configuracao
    x, t = sampling.sample_collocation(num_points = 10000) # 10000 pontos
    x_ic, t_ic = sampling.sample_initial(num_points = 10000)
    x_bc, t_bc = sampling.sample_boundary(num_points = 10000)

    # PINN values
    u = model(x,t)
    u_ic = model(x_ic,t_ic)
    u_bc = model(x_bc,x_bc)

    # True values
    residual = pde_fn((x, t), u)
    u_ic_true = ic_true_fn(x_ic) 
    u_bc_true = bc_true_fn(x_bc, t_bc)
    
    optimizer.zero_grad(set_to_none=True)
    
    loss_pde = criterion(residual, torch.zeros_like(residual))
    loss_ic = criterion(u_ic, u_ic_true)
    loss_bc = criterion(u_bc, u_bc_true)

    # Perda total ponderada
    loss = (loss_weights['pde'] * loss_pde + 
            loss_weights['ic'] * loss_ic + 
            loss_weights['bc'] * loss_bc)
    
    loss.backward()
    optimizer.step()

    if log_interval == 0:
        print(f"loss={loss.item():.4f} | pde={loss_pde:.4f}"
              f"IC={loss_ic.item():.4f} | BC={loss_bc:.4f}")

    return loss.item(), loss_pde.item(), loss_ic.item(), loss_bc.item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, target)

        running_loss += loss.item() * data.size(0)
        _, pred = output.max(1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    return running_loss / total, correct / total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Treino PINN com rastreabilidade de experimentos")
    parser.add_argument("--config", default="config/local.yaml",
                        help="Caminho do YAML de override (local.yaml ou server.yaml)")
    parser.add_argument("--default-config", default="config/default.yaml",
                        help="Caminho do YAML base")
    parser.add_argument("--name", default=None,
                        help="Sobrescreve experiment.name_prefix")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Sobrescreve epochs do config")
    parser.add_argument("--dry-run", action="store_true",
                        help="So valida config e constroi modelo, sem treinar")
    args = parser.parse_args()

    # --- Config (merge default + override) ---
    cfg = load_config(args.config, args.default_config)
    if args.name:
        cfg.setdefault("experiment", {})["name_prefix"] = args.name
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    # --- Initialization Writer - Tensorboard ---
    if cfg["tensorboard"]["enable"]:
        # Get current time to organize the folder's name for the tensorboard
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(
            f"./experiments/tensorboard/pinn/{cfg["tensorboard"]["name_prefix"]}_{timestamp}"
        )

    # --- Reprodutibilidade ---
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))

    # --- Device ---
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA solicitado mas indisponivel. Caindo para CPU.")
        device = torch.device("cpu")
    print(f"[Setup] Device: {device} | Epochs: {cfg['epochs']} | "
          f"LR: {cfg['optimizer']['learning_rate']}")

    # --- Experimento (cria pasta autocontida) ---
    exp = Experiment(
        base_dir=cfg["experiment_path"],
        cfg=cfg,
        name_prefix=cfg.get("experiment", {}).get("name_prefix", "run"),
    )

    # --- Redireciona stdout/stderr para o log do experimento ---
    log_file = open(exp.log_path, "a", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        print(f"[Experiment] name={exp.name}")
        print(f"[Experiment] dir={exp.dir}")
        print(f"[Experiment] config_hash={exp.cfg_hash}")
        print(f"[Experiment] git_commit={exp.git_info.get('commit')}")
        print(f"[Experiment] git_dirty={exp.git_info.get('is_dirty')}")
        if exp.git_info and exp.git_info.get("is_dirty"):
            print("[Experiment] ⚠️  Working tree sujo — reprodução exata requer o diff salvo em git_info.yaml")

        # --- Dados ---
        # Not required for PINNs. Check for specific cases
        #train_loader, test_loader = build_dataloaders(cfg)

        # --- PDE ---
        pde_fn, ic_true_fn, bc_true_fn = build_pde(cfg)

        # --- Data sampling ---
        data_sampling = SpatialTemporalDomain(cfg) 

        # --- Model ---
        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] {cfg['model']['name']} | params treináveis: {n_params:,}")

        if args.dry_run:
            print("[Dry-run] Config e modelo OK. Encerrando sem treinar.")
            exp.finalize(status="dry_run", n_params=n_params)
            return

        # --- Optimization ---
        criterion = build_criterion(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        # --- Loop de treino ---
        best_loss = np.inf
        start_time = time.time()

        for epoch in range(1, cfg["epochs"] + 1):
            epoch_start = time.time()
            print(f"\n===== Epoch {epoch}/{cfg['epochs']} =====")

            loss, pde_loss, ic_loss, bc_loss = train_one_epoch(
                pde_fn, ic_true_fn, bc_true_fn, cfg['pde']['params'],
                model, data_sampling, optimizer, criterion, cfg['loss_weights'], device,
                log_interval=cfg.get("log_interval", 0),
            )

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            print(f"[Epoch {epoch}] "
                  f"pde_loss={loss:.4f} pde_loss={pde_loss:.4f} | "
                  f"IC_loss={ic_loss:.4f} BC_loss={bc_loss:.4f} | "
                  f"lr={current_lr:.6f} | Epoch time={epoch_time:.1f}s")

            # Metricas (JSONL)
            exp.log_metrics({
                "epoch": epoch,
                "loss": loss,
                "pde_loss": pde_loss,
                "ic_loss": ic_loss,
                "bc_loss": bc_loss,
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 3),
            })

            # --- Collect Writer - Tensorboard ---
            if cfg["tensorboard"]["enable"]:
                # Writer (to be used in Tensorboard)
                writer.add_scalar("Loss/Total", loss, epoch)
                writer.add_scalar("Loss/PDE", pde_loss, epoch)
                writer.add_scalar("Loss/IC", ic_loss, epoch)
                writer.add_scalar("Loss/BC", bc_loss, epoch)

            # Checkpoint "last"
            if cfg.get("save_last", True):
                exp.save_checkpoint(
                    model.state_dict(),
                    name="last.pth",
                    extra_meta={"epoch": epoch, "loss": loss},
                )

            # Checkpoint "best"
            if cfg.get("save_best", True) and loss < best_loss:
                best_loss = loss
                exp.save_checkpoint(
                    model.state_dict(),
                    name="best.pth",
                    extra_meta={"epoch": epoch, "loss": loss,
                                "is_best": True},
                )
                print(f"  [Checkpoint] Novo best: loss={best_loss:.4f}")

            # Checkpoint por epoca (opcional)
            if cfg.get("save_every_epoch", False):
                exp.save_checkpoint(
                    model.state_dict(),
                    name=f"epoch_{epoch:02d}.pth",
                    extra_meta={"epoch": epoch, "loss": loss},
                )

            # Scheduler
            if scheduler is not None:
                scheduler.step()

        total_time = time.time() - start_time
        print(f"\n[Treino] Concluido em {total_time:.1f}s | best_loss={best_loss:.4f}")

        # --- Close Writer - Tensorboard ---
        if cfg["tensorboard"]["enable"]:
            writer.close()  

        exp.finalize(
            status="completed",
            best_loss=best_loss,
            epochs=cfg["epochs"],
            total_time_s=round(total_time, 2),
            n_params=n_params,
            device=str(device),
        )

    except Exception as e:
        print(f"\n[ERRO] Treino interrompido: {e}")
        exp.finalize(status="failed", error=str(e))
        raise
    finally:
        log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"[Experiment] Artefatos em: {exp.dir}")


if __name__ == "__main__":
    main()