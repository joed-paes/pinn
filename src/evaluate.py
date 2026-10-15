"""
Checkpoint evaluation salved by Experiment.

Uso:
    # Avaliar os experimentos de uma pasta específica
    python src/evaluate.py --experiments-dir experiments/

    # Avaliar um experimento específico que não foi avaliado
    python src/evaluate.py --experiment experiments/2026-.../checkpoints/best.pth

    # Avaliar todos os experimentos que ainda não foram avaliados 
    python src/evaluate.py --experiment experiments/2026-.../checkpoints/best.pth

    #  Ignora se já existe avaliação e refaz toda a análise
    python src/evaluate.py --ignore --all

    # Avaliar usando GPU
    python src/evaluate.py ... --device cuda

    # Avaliar o experimento mais recente automaticamente de uma pasta específica
    python src/evaluate.py --latest --experiments-dir experiments/

    # Salvar relatório JSON na pasta do experimento
    python src/evaluate.py --experiment ... --save-report

    # Avaliar o melhor .pth do último experimento, ignora se já existe avaliação e salva o report
    python src/evaluate.py --best --latest --ignore --save-report

    Codigo simples para checar um checkpoint:
    
    import torch
    ckpt = torch.load("experiments/.../checkpoints/best.pth", map_location="cpu")
    print("Config:", ckpt["config"])
    print("Git commit:", ckpt["git_commit"])
    print("Dirty:", ckpt["git_dirty"])

"""

import os
import glob
import yaml
import argparse
import json
import sys
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
# Configura o matplotlib para rodar em modo "headless" (servidor Linux via SSH)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Importa do pacote src
try:
    from src.train import build_criterion
    from src.model import build_model
    from src.pdes import build_pde
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import build_criterion
    from model import build_model
    from pdes import build_pde


# --------------------------------------------------------------------------- #
# Valid Checkpoint search
# --------------------------------------------------------------------------- #
def search_checkpoints(experiments_dir: Path, args: argparse.Namespace):
    valid_experiments = []
    
    if args.best:
        search_pattern = "best.pth"
    else:
        search_pattern = "*.pth"
    
    # Verification 1 - run last experiment
    if args.latest:
        experiment_folder = find_latest_checkpoint(experiments_dir)
        checkpoint_folder = experiment_folder / "checkpoints"
        validation_folder = experiment_folder / "graphic_validation"
        if validation_folder.exists() and validation_folder.is_dir() and not args.ignore:
            print(f"Validation folder already exists.")
            return []
        else:
            pth_files = list(checkpoint_folder.glob(search_pattern)) # just looking for best case ("*.pth" looks for all)
            if pth_files:
                valid_experiments.append({
                    "experiment_folder": experiment_folder,
                    "experiment": experiment_folder.name,
                    "checkpoint_path": checkpoint_folder,
                    "num_pth": len(pth_files),
                    "files": [str(checkpoint_folder / f.name) for f in pth_files]
                })
            return valid_experiments
    
    # Verification 2 - run specific experiment
    if args.experiment:
        experiment_folder = Path(args.experiment)
        checkpoint_folder = experiment_folder / "checkpoints"
        validation_folder = experiment_folder / "graphic_validation"
        if validation_folder.exists() and validation_folder.is_dir() and not args.ignore:
            print(f"Validation folder already exists.")
            return []
        else:
            pth_files = list(checkpoint_folder.glob(search_pattern)) # just looking for best case ("*.pth" looks for all)
            if pth_files:
                valid_experiments.append({
                    "experiment_folder": experiment_folder,
                    "experiment": experiment_folder.name,
                    "checkpoint_path": checkpoint_folder,
                    "num_pth": len(pth_files),
                    "files": [str(checkpoint_folder / f.name) for f in pth_files]
                })
            return valid_experiments

    # Verification 2 - run for all folders (if not validation already created)
    if args.all:
        # --- Loop for all checkpoints ---
        for checkpoint_folder in experiments_dir.glob("*/checkpoints"):
            experiment_folder = checkpoint_folder.parent

            # Filter only the valid experiments folder (starts with "2026..."")
            if re.match(r"^\d{4}", experiment_folder.name):
                
                validation_folder = experiment_folder / "graphic_validation"
                if validation_folder.exists() and validation_folder.is_dir() and not args.ignore:
                    continue # skip this experiment and moves next
                else:
                    pth_files = list(checkpoint_folder.glob(search_pattern))
                    if pth_files:
                        valid_experiments.append({
                            "experiment_folder": experiment_folder,
                            "experiment": experiment_folder.name,
                            "checkpoint_path": checkpoint_folder,
                            "num_pth": len(pth_files),
                            "files": [str(checkpoint_folder / f.name) for f in pth_files]
                        })
    
        print(f"\nFound {len(valid_experiments)} experiments with checkpoints to be evaluated.\n")
        return valid_experiments

    raise RuntimeError("Provide --all, --latest or --experiment arguments")

# --------------------------------------------------------------------------- #
# Get latest Checkpoint
# --------------------------------------------------------------------------- #
def find_latest_checkpoint(experiments_dir: Path) -> Path:
    """Encontra o best.pth do experimento mais recente."""
    runs = sorted(experiments_dir.glob("*/metadata.json"))
    if not runs:
        raise FileNotFoundError(f"Nenhum experimento encontrado em {experiments_dir}")
    latest = runs[-1].parent
    pths = list((latest / "checkpoints").glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"Nenhum checkpoint em {latest}")
    return latest

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
        "name": os.path.basename(path),
        "epoch": ckpt.get("epoch"),
        "loss": ckpt.get("loss"),
        "is_best": ckpt.get("is_best", False),
    }
    return model, cfg, meta

# --------------------------------------------------------------------------- #
# Run checkpoint
# --------------------------------------------------------------------------- #
def run_checkpoint(model, cfg, meta, OUTPUT_IMG_DIR):
    # --- Load geomentry ---
    x_min = cfg.get("pde", {}).get("params", {}).get("domain", {}).get('x_min', -1.0)
    x_max = cfg.get("pde", {}).get("params", {}).get("domain", {}).get('x_max', 1.0)
    t_min = cfg.get("pde", {}).get("params", {}).get("domain", {}).get('t_min', 0.0)
    t_max = cfg.get("pde", {}).get("params", {}).get("domain", {}).get('t_max', 1.0)
    # --- PDE ---
    nu = cfg.get("pde", {}).get("params", {}).get('nu', 0.01 / np.pi)
    pde_fn, _, _ = build_pde(cfg)

    print(f"-> Configurações carregadas do experimento: {meta["name"]}")
    print(f"-> Domínio Espacial: [{x_min}, {x_max}] | Domínio Temporal: [{t_min}, {t_max}]")

    # --- Device ---
    device = cfg.get("device")

    # --- PREPARAÇÃO DA MALHA DE TESTE FINA (Sem necessidade de técnica de amostragem) ---
    # Resolução alta (ex: 200x100) para garantir uma avaliação rigorosa da física
    n_x = cfg.get("num_spatial_points", 200)
    n_t = cfg.get("num_time_points", 100)
    x_space = np.linspace(x_min, x_max, n_x)
    t_space = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x_space, t_space)

    x_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32, requires_grad=True).to(device)
    t_tensor = torch.tensor(T.flatten()[:, None], dtype=torch.float32, requires_grad=True).to(device)

    # Inferência da solução aproximada u(x, t)
    u_pred_tensor = model(x_tensor, t_tensor)
    
    # Executa a função de cálculo de resíduo importada do seu src/pde.py
    # Adapte os argumentos abaixo de acordo com a assinatura exata da sua função calcular_residuo
    f_residuo_tensor = pde_fn((x_tensor, t_tensor), u_pred_tensor)
    
    # Conversão para NumPy para métricas e plotagem
    u_pred = u_pred_tensor.detach().cpu().numpy().reshape(X.shape)
    f_residuo = f_residuo_tensor.detach().cpu().numpy().reshape(X.shape)
    
    mse_physics = np.mean(f_residuo ** 2)
    max_residuo = np.max(np.abs(f_residuo))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot u(x,t)
    im1 = ax[0].imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                    extent=[t_space.min(), t_space.max(), x_space.min(), x_space.max()], 
                    origin='lower', aspect='auto')
    ax[0].set_title(f'Prediction $u(x,t)$\n{meta["name"]}')
    ax[0].set_xlabel('Time ($t$)')
    ax[0].set_ylabel('Space ($x$)')
    fig.colorbar(im1, ax=ax[0])
    
    # Plot do Resíduo Físico absoluto da EDP
    im2 = ax[1].imshow(np.abs(f_residuo), interpolation='nearest', cmap='viridis', 
                    extent=[t_space.min(), t_space.max(), x_space.min(), x_space.max()], 
                    origin='lower', aspect='auto')
    ax[1].set_title(f'PDE Physical Residual  $|f(x,t)|$\nMSE: {mse_physics:.2e}')
    ax[1].set_xlabel('Time ($t$)')
    ax[1].set_ylabel('Space ($x$)')
    fig.colorbar(im2, ax=ax[1])
    
    plt.tight_layout()
    
    # Salva o arquivo diretamente na pasta do experimento correspondente
    img_output_path = os.path.join(OUTPUT_IMG_DIR, f"{os.path.splitext(os.path.basename(meta["name"]))[0]}.png")
    plt.savefig(img_output_path, dpi=150)
    plt.close(fig)

    return {
        "num_spatial_points": n_x,
        "num_time_points": n_t,
        "mse_physics": mse_physics, 
        "max_residul": max_residuo
    }
# --------------------------------------------------------------------------- #
# Utilitarios de visualizacao
# --------------------------------------------------------------------------- #
def print_report(meta: dict, metrics: dict, idx: int):
    print("\n" + "=" * 60)
    print(f"Evaluation report - {idx+1}")
    print("=" * 60)

    print("\n--- Checkpoint ---")
    print(f"  config_hash : {meta.get('config_hash')}")
    print(f"  git_commit  : {meta.get('git_commit')}")
    print(f"  git_dirty   : {meta.get('git_dirty')}")
    print(f"  saved_at    : {meta.get('saved_at')}")
    print(f"  epoch       : {meta.get('epoch')}")
    print(f"  is_best     : {meta.get('is_best')}")
    print(f"  loss        : {meta.get('loss')}")

    print("\n--- Métricas globais ---")
    print(f"  Evaluating points : {metrics['num_spatial_points']*metrics['num_time_points']}")
    print(f"  MSE (Physics)     : {metrics['mse_physics']:.4f}")
    print(f"  Max. residual     : {metrics['max_residul']:.4f}")

    print("=" * 60 + "\n")

# --------------------------------------------------------------------------- #
# Decoder for JSON format
# --------------------------------------------------------------------------- #
def json_encoder(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj) # Fallback para qualquer outro tipo não suportado


def main():
    parser = argparse.ArgumentParser(description="Checkpoint evaluation salved by Experiment")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to specific experiment")
    parser.add_argument("--all", action="store_true",
                        help="Look for all *.pth available from experiments without evaluation")
    parser.add_argument("--ignore", action="store_true",
                        help="Run for all *.pth available, ignoring if evaluation exists")
    parser.add_argument("--latest", action="store_true",
                        help="Run last *.pth generated")
    parser.add_argument("--best", action="store_true",
                        help="Run only best.pth generated")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Experiment base path")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device de avaliação")
    parser.add_argument("--save-report", action="store_true",
                        help="Salve eval_report.json in experiment folder")
       
    args = parser.parse_args()

    # Look for checkpoints in experiments
    ckpt_list = search_checkpoints(Path(args.experiments_dir), args)

    if not ckpt_list:
        print("All experiments already have validation results.")
        sys.exit(1)
    else:
        print(f"[CLI] Selected checkpoints: {len(ckpt_list)}")

    # --- Device ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, moving to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[CLI] Device: {device}")

    
    for ckpt in ckpt_list:
        report = []
        for i, ckpt_file in enumerate(ckpt["files"]):
            # --- Output for image storage ---
            OUTPUT_IMG_DIR = ckpt["experiment_folder"] / "graphic_validation"
            os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)      

            # --- Load Checkpoint ---
            model, cfg, meta = load_checkpoint(ckpt_file, device)
            # --- Run Checkpoint ---
            metrics = run_checkpoint(model, cfg, meta, OUTPUT_IMG_DIR)
        
            # Report
            print_report(meta, metrics, i)
            report.append({
                "checkpoint_path": str(ckpt["checkpoint_path"]),
                "checkpoint_meta": meta,
                "metrics": metrics,
            })

        # Salva relatório
        if args.save_report:
            out = ckpt["checkpoint_path"].parent / "eval_report.json"
            out.write_text(json.dumps(report, indent=2, default=json_encoder))
            print(f"[Report] Salvo em: {out}")
            

if __name__ == "__main__":
    main()