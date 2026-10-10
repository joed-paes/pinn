# src/experiment.py
import os
import json
import subprocess
import yaml
import hashlib
from datetime import datetime
from pathlib import Path


def _git(cmd):
    try:
        return subprocess.check_output(
            ["git"] + cmd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def capture_git_info():
    """Captura o estado completo do Git no momento da execução."""
    commit = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git(["status", "--porcelain"])
    diff = _git(["diff"]) if status else ""

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
        "diff": diff,
        "status": status,
    }


def config_hash(cfg: dict) -> str:
    """Hash determinístico da config, útil para comparações rápidas."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


class Experiment:
    """
    Gerencia uma pasta de experimento autocontida.
    Uso:
        exp = Experiment(base_dir="experiments", cfg=cfg)
        exp.save_checkpoint(model, name="best.pth")
        exp.log_metrics({"epoch": 1, "loss": 0.5, "acc": 0.97})
        exp.finalize(best_acc=0.985)
    """

    def __init__(self, base_dir, cfg, name_prefix=None):
        self.cfg = cfg
        self.git_info = capture_git_info()
        self.cfg_hash = config_hash(cfg)

        # Nome da pasta inclui timestamp + hiperparâmetros-chave
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = name_prefix or "run"
        suffix = self._build_suffix(cfg)
        self.name = f"{timestamp}_{prefix}{suffix}"

        self.dir = Path(base_dir) / self.name
        self.ckpt_dir = self.dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Persistir config e git_info imediatamente
        self._dump_yaml(self.dir / "config.resolved.yaml", cfg)
        self._dump_yaml(self.dir / "git_info.yaml", self.git_info)

        # Arquivo de métricas (append por época)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.metrics_path.touch()

        # Log de texto
        self.log_path = self.dir / "train.log"

        print(f"[Experiment] Pasta criada: {self.dir}")
        if self.git_info and self.git_info["is_dirty"]:
            print("[Experiment] ⚠️  Working tree sujo — commit hash não é suficiente para reproduzir!")

    @staticmethod
    def _build_suffix(cfg):
        """Gera um sufixo legível com hiperparâmetros-chave."""
        keys = ["learning_rate", "batch_size", "epochs", "model"]
        parts = [f"{k}={cfg[k]}" for k in keys if k in cfg]
        return "_" + "_".join(parts) if parts else ""

    @staticmethod
    def _dump_yaml(path, data):
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def save_checkpoint(self, state, name="last.pth", extra_meta=None):
        """Salva checkpoint com metadados embutidos (config + git)."""
        payload = {
            "state_dict": state,
            "config": self.cfg,
            "config_hash": self.cfg_hash,
            "git_commit": self.git_info["commit"],
            "git_dirty": self.git_info["is_dirty"],
            "saved_at": datetime.now().isoformat(),
        }
        if extra_meta:
            payload.update(extra_meta)

        path = self.ckpt_dir / name
        import torch
        torch.save(payload, path)
        print(f"[Experiment] Checkpoint salvo: {path}")

    def log_metrics(self, metrics: dict):
        """Append de metricas por epoca (formato JSON)."""
        metrics = {**metrics, 'timestamp': datetime.now().isoformat()}
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def finalize(self, **summary):
        """Grava o metadata.json final com um resumo do run."""
        meta = {
            "experiment_name": self.name,
            "config_hash": self.cfg_hash,
            "git_commit": self.git_info["commit"],
            "git_dirty": self.git_info["is_dirty"],
            "finished_at": datetime.now().isoformat(),
            **summary,
        }
        with open(self.dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Experiment] Finalizado: {self.dir}")

    def latest(cls, base_dir):
        runs = sorted(Path(base_dir).glob("*/metadata.json"))
        return runs[-1].parent if runs else None