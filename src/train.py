"""Walk-forward training loop for the GDL-Quantum-Finance model.

Orchestrates temporal walk-forward cross-validation with early
stopping, checkpoint saving, and per-fold evaluation.

Theoretical Context:
    Financial time series are non-stationary. Standard K-Fold CV introduces
    lookahead bias (training on future data). We use "Walk-Forward Validation"
    (Purvey, 2018; Aronson, 2006) to strictly enforce time-causality.
    The model trains on a rolling window [t, t+N] and validates on [t+N, t+N+M],
    simulating a real-world production trading system.

References:
    Purvey (2018) "Advances in Financial Machine Learning."
    Aronson (2006) "Evidence-Based Technical Analysis."

Usage:
    cd GDL-Quantum-Finance
    uv run python src/train.py                     # full run
    uv run python src/train.py --smoke-test        # synthetic data
    uv run python src/train.py --max-folds 2       # limit folds
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None

from tqdm import tqdm

# Enable CPU fallback for MPS-unsupported operations (e.g. eigvalsh).
# Must be set before torch import to take effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data_loader import (
    FinancialGraphDataset,
    WalkForwardSplitter,
    build_synthetic_dataframe,
    get_device,
)
from src.evaluation import evaluate_model_on_dataset
from src.layers.graph_utils import build_batched_graphs_gpu
from src.losses import GDLQuantumLoss
from src.models import GDLQuantumFinanceModel

logger = logging.getLogger(__name__)

# Project root and standard paths.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"
_CHECKPOINT_DIR = _PROJECT_ROOT / "output" / "checkpoints"


def _validate_device(device: torch.device) -> torch.device:
    """Validates that the chosen device supports all required operations.

    The quantum layer uses torch.linalg.solve (Cayley transform) whose
    backward pass has a known bug on MPS (IndexError in pivot
    permutation). Until PyTorch fixes this, training must use CPU or
    CUDA, with this function falling back to CPU when MPS is selected.

    Args:
        device: Candidate compute device.

    Returns:
        The validated device (CPU if MPS was selected).
    """
    if device.type == "mps":
        logger.warning(
            "MPS detected but torch.linalg.solve backward is broken on "
            "MPS for the Cayley transform (quantum layer). "
            "Falling back to CPU for training."
        )
        return torch.device("cpu")

    return device


def _configure_logging() -> None:
    """Configures the root logger with the project-standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: Path | None = None) -> dict:
    """Loads the YAML configuration file.

    Args:
        path: Path to config.yaml. Defaults to the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is absent.
    """
    path = path or _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s.", path)
    return cfg


def build_model(config: dict, device: torch.device) -> GDLQuantumFinanceModel:
    """Constructs the GDLQuantumFinanceModel from configuration.

    Args:
        config: Full configuration dictionary.
        device: Target compute device.

    Returns:
        Model instance placed on the specified device.
    """
    geo = config["model"]["geometric"]
    qm = config["model"]["quantum"]
    gr = config["model"]["graph"]
    pred = config["model"]["prediction"]

    model = GDLQuantumFinanceModel(
        feature_dim=geo["input_dim"],
        embedding_dim=geo["embedding_dim"],
        hidden_dim=gr["hidden_channels"],
        graph_out_dim=gr["out_channels"],
        num_regimes=qm["num_regimes"],
        gat_heads=gr["gat_heads"],
        context_dim=qm["context_dim"],
        prediction_hidden_dim=pred["hidden_dim"],
        dropout=gr.get("dropout", 0.1),
        edge_drop=gr.get("edge_drop", 0.1),
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model initialised on %s: %d trainable parameters.",
        device,
        n_params,
    )
    return model


def build_criterion(config: dict) -> GDLQuantumLoss:
    """Constructs the multi-objective loss function from configuration.

    Supports two configuration styles:
      1) Backward-compatible single profile:
         training:
           loss_weights:
             prediction: 1.0
             uniformity: 0.01
             density_matrix: 0.001
             directional: 0.10
      2) Named profiles for A/B/C testing:
         training:
           loss_profiles:
             A:
               prediction: 1.0
               uniformity: 0.01
               density_matrix: 0.001
               directional: 0.10
             B:
               prediction: 1.0
               uniformity: 0.005
               density_matrix: 0.0005
               directional: 0.20
             C:
               prediction: 1.0
               uniformity: 0.002
               density_matrix: 0.0002
               directional: 0.30
           loss_profile: "A"

    Args:
        config: Full configuration dictionary.

    Returns:
        GDLQuantumLoss instance with configured weights.
    """
    training_cfg = config.get("training", {})
    weights, _, _ = _resolve_loss_weights(config)
    uniformity_max_points = training_cfg.get("uniformity_max_points", 2048)
    return GDLQuantumLoss(
        w_pred=weights.get("prediction", 1.0),
        w_uniform=weights.get("uniformity", 0.01),
        w_dm=weights.get("density_matrix", 0.001),
        w_dir=weights.get("directional", 0.1),
        uniformity_max_points=uniformity_max_points,
    )


def _resolve_loss_weights(config: dict) -> tuple[dict[str, float], str, bool]:
    """Resolves active loss weights from named profiles or fallback weights.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (weights, active_profile_name, using_profiles).
    """
    training_cfg = config.get("training", {})
    profiles = training_cfg.get("loss_profiles", {})
    profile_name = training_cfg.get("loss_profile", "default")

    if isinstance(profiles, dict) and profile_name in profiles:
        selected_weights = profiles[profile_name]
        return selected_weights, str(profile_name), True

    fallback_weights = training_cfg.get("loss_weights", {})
    if isinstance(profiles, dict) and profiles and profile_name not in profiles:
        logger.warning(
            "Requested loss profile '%s' not found. Falling back to training.loss_weights.",
            profile_name,
        )
    return fallback_weights, "default", False


class EarlyStopping:
    """Monitors validation loss and signals when to stop training.

    Tracks the best observed validation loss and triggers stopping
    when no improvement is seen for a configurable number of epochs.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum decrease in loss to count as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Checks whether training should stop.

        Args:
            val_loss: Current epoch validation loss.
            epoch: Current epoch number (zero-based).

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.should_stop = False
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                "Early stopping triggered at epoch %d. " "Best epoch: %d (loss=%.6f).",
                epoch,
                self.best_epoch,
                self.best_loss,
            )
            return True
        return False


def _make_dataloader(
    dataset: FinancialGraphDataset,
    config: dict,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> PyGDataLoader:
    """Wraps a dataset in a PyG DataLoader with prefetching.

    Args:
        dataset: The financial graph dataset.
        config: Full configuration dictionary.
        shuffle: Whether to shuffle samples (False for validation).
        generator: Optional dedicated RNG for the shuffle order, so batch order
            does not depend on how many global-RNG draws model construction used.

    Returns:
        PyG DataLoader with configured batch size and workers.
    """
    t_cfg = config.get("training", {})
    batch_size = t_cfg.get("batch_size", 32)
    num_workers = t_cfg.get("dataloader_workers", 4)
    use_cuda = torch.cuda.is_available()

    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        drop_last=False,
        generator=generator,
    )


def _forward_step(
    model: nn.Module,
    batch_data: Any,
    criterion: GDLQuantumLoss,
    device: torch.device,
    build_graph: bool = True,
    graph_threshold: float = 0.0,
    max_neighbours: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Runs a single forward step on a mini-batch of cross-sections.

    Handles moving a PyG Batch object to the device, optionally
    building per-date correlation graphs on GPU, and computing the loss.

    Args:
        model: Any model conforming to the flat-format forward API.
        batch_data: PyG Batch object (multiple dates collated).
        criterion: The multi-objective loss function.
        device: Compute device.
        build_graph: If True, build correlation graphs from past_returns.
            Set False for configs that do not use the graph (A0, A1, A2, A4, A6, A9).
        graph_threshold: Minimum absolute correlation to retain an edge.
        max_neighbours: Maximum retained neighbours per node after
            thresholding.

    Returns:
        Tuple of (loss_dict, predictions, targets) where loss_dict
        contains all loss components and the total loss.
    """
    batch_data = batch_data.to(device)

    x = batch_data.x                     # (total_nodes, D)
    # Combined PyG connectivity (may be replaced below when build_graph is true).
    edge_index = batch_data.edge_index   # (2, E)
    edge_weight = batch_data.edge_attr   # (E,)
    context = batch_data.context          # (B, C)
    targets = batch_data.y                # (total_nodes,)
    target_mask = batch_data.target_mask  # (total_nodes,)
    batch_vec = batch_data.batch          # (total_nodes,)

    # Optional sequence data for LSTM baselines
    x_seq = getattr(batch_data, "x_seq", None)

    # Dynamic graph construction on GPU only when the model uses the graph.
    # Skipping this for A0, A1, A2, A4, A6, A9 saves substantial time per batch.
    if build_graph:
        past_returns = getattr(batch_data, "past_returns", None)
        if past_returns is not None:
            edge_index, edge_weight = build_batched_graphs_gpu(
                past_returns,
                batch_vec,
                threshold=graph_threshold,
                max_neighbours=max_neighbours,
            )

    # Model forward pass (fully vectorised, no Python batch loop)
    output = model(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        context=context,
        batch=batch_vec,
        x_seq=x_seq,
    )

    # Extract predictions for masked targets across the batch
    preds = output["predictions"].squeeze(-1)  # (total_nodes,)
    masked_preds = preds[target_mask]
    masked_targets = targets[target_mask]

    # Compute loss on all masked predictions at once
    loss_dict = criterion(
        masked_preds,
        masked_targets,
        output["embeddings"],
        output["density_matrix"],
        output["regime_probs"],
    )

    return loss_dict, masked_preds.detach(), masked_targets.detach()


def train_one_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
    criterion: GDLQuantumLoss,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_graph: bool = True,
    graph_threshold: float = 0.0,
    max_neighbours: int | None = None,
) -> dict[str, float]:
    """Runs one training epoch using a pre-existing DataLoader.

    Args:
        model: The model to train.
        loader: PyG DataLoader (already initialised with workers).
        criterion: Multi-objective loss function.
        optimiser: Parameter optimiser.
        device: Compute device.
        scaler: AMP GradScaler for mixed-precision (None to disable).
        use_graph: If False, skip correlation graph build.
        graph_threshold: Minimum absolute correlation to retain an edge.
        max_neighbours: Maximum retained neighbours per node after
            thresholding.

    Returns:
        Dictionary of average loss components across the epoch plus the
        number of optimiser updates that actually occurred.
    """
    model.train()
    accumulators: dict[str, list[float]] = {
        "total": [],
        "prediction": [],
        "uniformity": [],
        "density_matrix": [],
        "directional": [],
    }
    optimiser_steps = 0

    use_amp = scaler is not None

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_data in pbar:
        # Skip mini-batches with no valid targets
        if batch_data.target_mask.sum() == 0:
            continue

        optimiser.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss_dict, _, _ = _forward_step(
                    model,
                    batch_data,
                    criterion,
                    device,
                    build_graph=use_graph,
                    graph_threshold=graph_threshold,
                    max_neighbours=max_neighbours,
                )
        else:
            loss_dict, _, _ = _forward_step(
                model,
                batch_data,
                criterion,
                device,
                build_graph=use_graph,
                graph_threshold=graph_threshold,
                max_neighbours=max_neighbours,
            )

        total_val = loss_dict["total"]

        # Guard: skip steps where the loss is NaN or inf
        if not torch.isfinite(total_val):
            continue

        if use_amp:
            scaler.scale(total_val).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                optimiser_steps += 1
        else:
            total_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser_steps += 1

        current_loss = total_val.item()
        for key in accumulators:
            accumulators[key].append(loss_dict[key].item())

        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    metrics: dict[str, float] = {}
    for key, values in accumulators.items():
        metrics[key] = float(np.mean(values)) if values else 0.0
    metrics["optimiser_steps"] = float(optimiser_steps)

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
    criterion: GDLQuantumLoss,
    device: torch.device,
    use_graph: bool = True,
    graph_threshold: float = 0.0,
    max_neighbours: int | None = None,
) -> dict[str, float]:
    """Runs one validation pass using a pre-existing DataLoader.

    Args:
        model: The model to evaluate.
        loader: PyG DataLoader.
        criterion: Multi-objective loss function.
        device: Compute device.
        use_graph: If False, skip correlation graph build.
        graph_threshold: Minimum absolute correlation to retain an edge.
        max_neighbours: Maximum retained neighbours per node after
            thresholding.

    Returns:
        Dictionary of average loss components across the epoch.
    """
    model.eval()
    accumulators: dict[str, list[float]] = {
        "total": [],
        "prediction": [],
        "uniformity": [],
        "density_matrix": [],
        "directional": [],
    }

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch_data in pbar:
        if batch_data.target_mask.sum() == 0:
            continue

        loss_dict, _, _ = _forward_step(
            model,
            batch_data,
            criterion,
            device,
            build_graph=use_graph,
            graph_threshold=graph_threshold,
            max_neighbours=max_neighbours,
        )

        total_val = loss_dict["total"]
        if not torch.isfinite(total_val):
            continue

        current_loss = total_val.item()
        for key in accumulators:
            accumulators[key].append(loss_dict[key].item())

        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    metrics: dict[str, float] = {}
    for key, values in accumulators.items():
        metrics[key] = float(np.mean(values)) if values else 0.0

    return metrics


def evaluate_fold(
    model: GDLQuantumFinanceModel,
    dataset: FinancialGraphDataset,
    device: torch.device,
) -> dict[str, float]:
    """Computes evaluation metrics on a validation fold.

    Delegates to ``evaluation.evaluate_model_on_dataset`` for the full
    metric suite (MSE, MAE, Directional Accuracy, IC, Sharpe, Max
    Drawdown).  This thin wrapper keeps the ``train_fold`` call-site
    unchanged.

    Args:
        model: Trained model to evaluate.
        dataset: Validation dataset (date-restricted subset).
        device: Compute device.

    Returns:
        Dictionary of named evaluation metrics.
    """
    return evaluate_model_on_dataset(model, dataset, device)


def _save_checkpoint(
    model: GDLQuantumFinanceModel,
    fold: int,
    epoch: int,
    val_loss: float,
    metrics: dict[str, float],
    checkpoint_dir: Path,
) -> Path:
    """Saves a model checkpoint to disc.

    Args:
        model: Model whose state to save.
        fold: Fold index.
        epoch: Best epoch number.
        val_loss: Best validation loss achieved.
        metrics: Evaluation metrics at best epoch.
        checkpoint_dir: Directory for checkpoint files.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"fold_{fold:02d}_best.pt"

    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
        },
        path,
    )
    logger.info(
        "Checkpoint saved: %s (epoch=%d, val_loss=%.6f).", path, epoch, val_loss
    )
    return path


def train_fold(
    fold_idx: int,
    train_dates: list[pd.Timestamp],
    val_dates: list[pd.Timestamp],
    config: dict,
    device: torch.device,
    parquet_path: Path | None = None,
    dataframe: pd.DataFrame | None = None,
    use_wandb: bool = False,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Trains and evaluates the model on a single walk-forward fold.

    Creates fresh datasets, model, optimiser, and scheduler for the
    fold, then trains for up to max_epochs with early stopping. Saves
    the best checkpoint and returns evaluation metrics.

    Args:
        fold_idx: Zero-based fold index.
        train_dates: Training date window.
        val_dates: Validation date window.
        config: Full configuration dictionary.
        device: Compute device.
        parquet_path: Path to fact_table.parquet (optional).
        dataframe: Pre-loaded DataFrame for smoke testing.
        use_wandb: Whether to log to Weights & Biases.
        experiment_id: Optional ID to group folds in W&B.

    Returns:
        Dictionary with fold results including metrics and timing.
    """
    if use_wandb:
        try:
            w_cfg = config.get("wandb", {})
            wandb.init(
                project=w_cfg.get("project", "gdl-quantum-finance"),
                entity=w_cfg.get("entity"),
                tags=w_cfg.get("tags", []),
                mode=w_cfg.get("mode", "online"),
                group=experiment_id,
                name=f"fold_{fold_idx:02d}",
                config=config,
                reinit=True,
            )
        except Exception as e:
            logger.error("Failed to initialise W&B: %s. Continuing without logging.", e)
            use_wandb = False

    t_cfg = config["training"]
    max_epochs: int = t_cfg["max_epochs"]
    patience: int = t_cfg["early_stopping_patience"]
    min_delta: float = t_cfg.get("min_delta", 0.0)

    logger.info(
        "Fold %d: training %d dates, validating %d dates.",
        fold_idx,
        len(train_dates),
        len(val_dates),
    )

    # Build datasets restricted to the fold's date windows
    common_kwargs: dict[str, Any] = {"config": config}
    if dataframe is not None:
        common_kwargs["dataframe"] = dataframe
        common_kwargs["parquet_path"] = "unused"
    elif parquet_path is not None:
        common_kwargs["parquet_path"] = parquet_path
    else:
        common_kwargs["parquet_path"] = _PROJECT_ROOT / config["data"]["parquet_path"]

    train_ds = FinancialGraphDataset(
        dates=train_dates,
        **common_kwargs,
    )
    val_ds = FinancialGraphDataset(
        dates=val_dates,
        **common_kwargs,
    )

    logger.info(
        "Fold %d datasets: train=%d dates, val=%d dates.",
        fold_idx,
        len(train_ds),
        len(val_ds),
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.warning(
            "Fold %d has empty dataset (train=%d, val=%d). Skipping.",
            fold_idx,
            len(train_ds),
            len(val_ds),
        )
        if use_wandb:
            wandb.finish()
        return {"fold": fold_idx, "skipped": True}

    # Initialise model, loss, optimiser, scheduler
    model = build_model(config, device)
    # Intent: Multi-Objective Optimisation.
    # We don't just minimise MSE. We simultaneously optimise for:
    # 1. Prediction Accuracy (MSE)
    # 2. Geometric Structure (Uniformity Loss)
    # 3. Regime Diversity (Entropy Regularisation)
    # 4. Financial Utility (Directional Loss)
    # This aligns the neural network with the theoretical constraints of the layers.
    active_weights, active_profile, using_profiles = _resolve_loss_weights(config)
    criterion = build_criterion(config)
    logger.info(
        "Fold %d using loss profile '%s' with weights: %s",
        fold_idx,
        active_profile if using_profiles else "default",
        active_weights,
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max_epochs,
        eta_min=t_cfg["learning_rate"] * 0.01,
    )

    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)

    # AMP (mixed-precision) setup for CUDA tensor cores
    use_amp = t_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    graph_cfg = config.get("model", {}).get("graph_construction", {})
    graph_threshold = float(graph_cfg["correlation_threshold"])
    max_neighbours_cfg = graph_cfg.get("max_neighbours")
    max_neighbours = int(max_neighbours_cfg) if max_neighbours_cfg is not None else None
    if use_amp:
        logger.info("Fold %d: mixed-precision training enabled (FP16).", fold_idx)

    # Build DataLoaders once (workers persist across epochs)
    train_loader = _make_dataloader(train_ds, config, shuffle=True)
    val_loader = _make_dataloader(val_ds, config, shuffle=False)

    # Training loop
    best_model_state: dict | None = None
    fold_start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            device,
            scaler=scaler,
            graph_threshold=graph_threshold,
            max_neighbours=max_neighbours,
        )

        # Validate
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            graph_threshold=graph_threshold,
            max_neighbours=max_neighbours,
        )
        current_lr = optimiser.param_groups[0]["lr"]

        # Log epoch summary to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics["total"],
                "train/prediction": train_metrics["prediction"],
                "train/uniformity": train_metrics["uniformity"],
                "train/density_matrix": train_metrics["density_matrix"],
                "train/directional": train_metrics["directional"],
                "val/loss": val_metrics["total"],
                "val/prediction": val_metrics["prediction"],
                "learning_rate": current_lr,
            })

        epoch_time = time.time() - epoch_start

        logger.info(
            "Fold %d | Epoch %3d/%d | "
            "Train loss=%.6f (pred=%.6f, uni=%.4f, dm=%.6f, dir=%.4f) | "
            "Val loss=%.6f (pred=%.6f) | "
            "LR=%.2e | %.1fs",
            fold_idx,
            epoch + 1,
            max_epochs,
            train_metrics["total"],
            train_metrics["prediction"],
            train_metrics["uniformity"],
            train_metrics["density_matrix"],
            train_metrics["directional"],
            val_metrics["total"],
            val_metrics["prediction"],
            current_lr,
            epoch_time,
        )

        # Early stopping check
        if early_stop(val_metrics["total"], epoch):
            break

        # Save best model state in memory
        if early_stop.best_epoch == epoch:
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if train_metrics["optimiser_steps"] > 0:
            scheduler.step()
        else:
            logger.warning(
                "Fold %d | Epoch %d: skipping scheduler.step() because no "
                "optimiser step occurred.",
                fold_idx,
                epoch + 1,
            )

    fold_time = time.time() - fold_start
    logger.info(
        "Fold %d training complete: %d epochs in %.1fs. "
        "Best epoch: %d (val_loss=%.6f).",
        fold_idx,
        early_stop.best_epoch + 1,
        fold_time,
        early_stop.best_epoch + 1,
        early_stop.best_loss or float("nan"),
    )

    # Restore best model weights for evaluation
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Evaluate on validation set
    eval_metrics = evaluate_fold(model, val_ds, device)

    logger.info(
        "INFO: Fold %d: IC=%.4f, DirAcc=%.2f%%, Sharpe(daily)=%.2f, "
        "Sharpe(annual)=%.2f, MSE=%.6f, MaxDD=%.4f.",
        fold_idx + 1,
        eval_metrics["ic"],
        eval_metrics["directional_accuracy"],
        eval_metrics["sharpe_daily"],
        eval_metrics["sharpe_annual"],
        eval_metrics["mse"],
        eval_metrics.get("max_drawdown", float("nan")),
    )

    # Save checkpoint
    _save_checkpoint(
        model=model,
        fold=fold_idx,
        epoch=early_stop.best_epoch,
        val_loss=early_stop.best_loss or float("nan"),
        metrics=eval_metrics,
        checkpoint_dir=_CHECKPOINT_DIR,
    )

    if use_wandb:
        # Log final evaluation metrics
        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
        wandb.finish()

    return {
        "fold": fold_idx,
        "skipped": False,
        "best_epoch": early_stop.best_epoch + 1,
        "best_val_loss": early_stop.best_loss,
        "training_time_s": fold_time,
        "metrics": eval_metrics,
    }


def run_training(
    config: dict,
    smoke_test: bool = False,
    max_folds: int | None = None,
    device_override: str | None = None,
    use_wandb: bool = False,
) -> list[dict[str, Any]]:
    """Orchestrates walk-forward training across all temporal folds.

    This is the main entry point for the training pipeline. It loads
    (or synthesises) data, constructs the walk-forward splitter, and
    iterates through each fold.

    Args:
        config: Full configuration dictionary.
        smoke_test: If True, uses synthetic data for quick validation.
        max_folds: Maximum number of folds to train (None = all).
        device_override: Force a specific device (e.g. "cpu", "cuda").
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        List of per-fold result dictionaries.
    """
    if device_override is not None:
        device = torch.device(device_override)
        logger.info("Device override: %s.", device)
    else:
        raw_device = get_device()
        device = _validate_device(raw_device)

    # Unique experiment ID for grouping folds
    experiment_id = f"exp_{int(time.time())}" if use_wandb else None

    parquet_path = _PROJECT_ROOT / config["data"]["parquet_path"]

    # Resolve data source: real parquet or synthetic smoke test
    dataframe: pd.DataFrame | None = None
    if smoke_test or not parquet_path.exists():
        if not smoke_test:
            logger.warning(
                "Parquet file not found at %s. "
                "Falling back to synthetic smoke-test data.",
                parquet_path,
            )
        logger.info("Generating synthetic data for smoke test.")
        dataframe = build_synthetic_dataframe(
            n_tickers=20,
            n_dates=400,
            seed=42,
        )

    # Build the full dataset to obtain the universe of valid dates
    if dataframe is not None:
        full_ds = FinancialGraphDataset(
            parquet_path="unused",
            config=config,
            dataframe=dataframe,
        )
    else:
        full_ds = FinancialGraphDataset(
            parquet_path=parquet_path,
            config=config,
        )

    # Intent: Time-Causality (Aronson, 2006).
    # We never shuffle financial data. We split strictly by time to ensure
    # the model only sees past data during training, preventing
    # "Oracle" lookahead bias.
    # Walk-forward splitter
    wf_cfg = config["training"]["walk_forward"]
    splitter = WalkForwardSplitter(
        dates=full_ds.valid_dates,
        train_window=wf_cfg["train_window"],
        validation_window=wf_cfg["validation_window"],
        step_size=wf_cfg["step_size"],
    )

    if len(splitter) == 0:
        logger.warning(
            "No valid folds available. The dataset has %d dates but "
            "requires %d (train=%d + val=%d). "
            "Reducing windows for smoke test.",
            len(full_ds.valid_dates),
            wf_cfg["train_window"] + wf_cfg["validation_window"],
            wf_cfg["train_window"],
            wf_cfg["validation_window"],
        )
        # For smoke tests with insufficient dates, create a minimal split.
        n_valid = len(full_ds.valid_dates)
        if n_valid >= 4:
            split_point = max(2, n_valid * 3 // 4)
            train_dates = full_ds.valid_dates[:split_point]
            val_dates = full_ds.valid_dates[split_point:]
            logger.info(
                "Smoke-test fallback split: train=%d, val=%d dates.",
                len(train_dates),
                len(val_dates),
            )

            # Override max_epochs for smoke test
            smoke_config = _deep_copy_config(config)
            smoke_config["training"]["max_epochs"] = min(
                3,
                config["training"]["max_epochs"],
            )

            result = train_fold(
                fold_idx=0,
                train_dates=train_dates,
                val_dates=val_dates,
                config=smoke_config,
                device=device,
                dataframe=dataframe,
                use_wandb=use_wandb,
                experiment_id=experiment_id,
            )
            return [result]

        logger.error("Insufficient data even for smoke test (%d dates).", n_valid)
        return []

    # Iterate through walk-forward folds
    n_folds = max_folds if max_folds is not None else len(splitter)
    n_folds = min(n_folds, len(splitter))
    logger.info("Starting training: %d folds.", n_folds)

    fold_results: list[dict[str, Any]] = []
    for fold_idx, (train_dates, val_dates) in enumerate(splitter):
        if fold_idx >= n_folds:
            break

        result = train_fold(
            fold_idx=fold_idx,
            train_dates=train_dates,
            val_dates=val_dates,
            config=config,
            device=device,
            parquet_path=parquet_path if dataframe is None else None,
            dataframe=dataframe,
            use_wandb=use_wandb,
            experiment_id=experiment_id,
        )
        fold_results.append(result)

    # Summary
    _log_summary(fold_results)
    return fold_results


def _deep_copy_config(config: dict) -> dict:
    """Creates a deep copy of a nested configuration dictionary.

    Args:
        config: Original configuration.

    Returns:
        Independent copy with no shared references.
    """
    import copy

    return copy.deepcopy(config)


def _log_summary(results: list[dict[str, Any]]) -> None:
    """Logs an aggregate summary of all fold results.

    Args:
        results: List of per-fold result dictionaries.
    """
    completed = [r for r in results if not r.get("skipped", False)]
    if not completed:
        logger.warning("No folds completed successfully.")
        return

    metrics_keys = [
        "ic",
        "directional_accuracy",
        "mse",
        "mae",
        "sharpe_daily",
        "sharpe_annual",
        "max_drawdown",
    ]
    logger.info("Training complete: %d/%d folds.", len(completed), len(results))

    for key in metrics_keys:
        values = [
            r["metrics"][key]
            for r in completed
            if key in r.get("metrics", {}) and np.isfinite(r["metrics"][key])
        ]
        if values:
            logger.info(
                "  %s: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                key,
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
            )

    total_time = sum(r.get("training_time_s", 0.0) for r in completed)
    logger.info("  Total training time: %.1fs.", total_time)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="GDL-Quantum-Finance walk-forward training loop.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: project root).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with synthetic data for quick end-to-end validation.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Maximum number of walk-forward folds to train.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config.",
    )
    parser.add_argument(
        "--loss-profiles",
        type=str,
        default=None,
        help=(
            "Comma-separated loss profile names for sensitivity runs "
            "(for example: A,B,C)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force a specific compute device.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (overrides config mode).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the training script."""
    _configure_logging()
    args = parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Apply command-line overrides
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
        logger.info("Overriding max_epochs to %d.", args.max_epochs)

    # Determine W&B status
    # Priority: CLI flags > Config file > Default
    config_wandb = config.get("wandb", {})
    mode = config_wandb.get("mode", "online")

    if args.no_wandb:
        use_wandb = False
        config.setdefault("wandb", {})["mode"] = "disabled"
    elif args.wandb:
        use_wandb = True
        config.setdefault("wandb", {})["mode"] = "online"
    else:
        # Default behavior: if smoke_test, disable unless configured otherwise
        if args.smoke_test:
            use_wandb = False
            logger.info("Smoke test: W&B disabled by default (use --wandb to enable).")
        else:
            use_wandb = (mode == "online")

    # If W&B is not installed, force disable
    if wandb is None:
        if use_wandb:
            logger.warning("Weights & Biases is not installed. Disabling logging.")
        use_wandb = False

    logger.info(
        "GDL-Quantum-Finance Training | smoke_test=%s | max_folds=%s | wandb=%s",
        args.smoke_test,
        args.max_folds,
        use_wandb,
    )
    requested_profiles: list[str] = []
    if args.loss_profiles:
        requested_profiles = [
            profile.strip() for profile in args.loss_profiles.split(",") if profile.strip()
        ]

    if requested_profiles:
        logger.info("Running loss-profile sensitivity for profiles: %s", requested_profiles)
        profile_summaries: dict[str, dict[str, float]] = {}
        for profile_name in requested_profiles:
            profile_config = _deep_copy_config(config)
            profile_config.setdefault("training", {})["loss_profile"] = profile_name
            logger.info("=== Running training with loss_profile=%s ===", profile_name)

            profile_results = run_training(
                config=profile_config,
                smoke_test=args.smoke_test,
                max_folds=args.max_folds,
                device_override=args.device,
                use_wandb=use_wandb,
            )
            if not profile_results:
                logger.error(
                    "Training produced no results for loss_profile=%s.",
                    profile_name,
                )
                continue

            profile_summaries[profile_name] = _summarise_sensitivity_metrics(
                profile_results
            )
            logger.info(
                "Finished loss_profile=%s. Fold count: %d",
                profile_name,
                len(profile_results),
            )

        if profile_summaries:
            logger.info("Loss-profile sensitivity summary (validation metrics):")
            for profile_name, summary in profile_summaries.items():
                logger.info(
                    "  %s | IC=%.6f | DirectionalAccuracy=%.6f | SharpeDaily=%.6f",
                    profile_name,
                    summary["ic"],
                    summary["directional_accuracy"],
                    summary["sharpe_daily"],
                )
    else:
        results = run_training(
            config=config,
            smoke_test=args.smoke_test,
            max_folds=args.max_folds,
            device_override=args.device,
            use_wandb=use_wandb,
        )

        if not results:
            logger.error("Training produced no results.")
        else:
            logger.info("Training pipeline finished. %d fold(s) processed.", len(results))


def _summarise_sensitivity_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Computes mean IC, directional accuracy, and Sharpe across folds.

    Args:
        results: Per-fold result dictionaries from run_training.

    Returns:
        Dictionary with mean values for the three sensitivity metrics.
    """
    completed = [result for result in results if not result.get("skipped", False)]
    if not completed:
        return {
            "ic": float("nan"),
            "directional_accuracy": float("nan"),
            "sharpe_daily": float("nan"),
        }

    def _mean_metric(metric_name: str) -> float:
        metric_values = [
            fold["metrics"][metric_name]
            for fold in completed
            if metric_name in fold.get("metrics", {})
            and np.isfinite(fold["metrics"][metric_name])
        ]
        if not metric_values:
            return float("nan")
        return float(np.mean(metric_values))

    return {
        "ic": _mean_metric("ic"),
        "directional_accuracy": _mean_metric("directional_accuracy"),
        "sharpe_daily": _mean_metric("sharpe_daily"),
    }


if __name__ == "__main__":
    main()
