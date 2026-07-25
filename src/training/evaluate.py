"""Evaluation stage for TimeTensor experiments."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from dataset import fetch_training_data, get_sizes
from models import load_model
from models.normalizations import get_normal_stats
from runtime import (
    batch_size,
    dataset_path,
    default_sampling,
    default_splits,
    default_subsets,
    device,
    model_specs,
    normalization_needs_stats,
    pretrained_path,
    run_dir,
    save_torch,
    section,
    seed,
    setup_logging,
    stats_eps,
    stats_max_windows,
    stats_seed,
    task_shape,
    to_plain_config,
)

from .losses import get_losses
from .pipeline import TorchLearner


LOGGER = logging.getLogger(__name__)


def _fetch_loaders(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    lags, horizon = task_shape(config)
    data_cfg = section(config, "data")
    return fetch_training_data(
        dataset_path(config),
        default_splits(config),
        default_sampling(config),
        default_subsets(config),
        batch_size(config),
        lags,
        horizon,
        seed=seed(config),
        stats_save_path=None,
        compute_stats=normalization_needs_stats(config),
        stats_max_windows=stats_max_windows(config),
        stats_seed=stats_seed(config),
        stats_eps=stats_eps(config),
        legacy_context_kind=data_cfg.get("legacy_context_kind"),
    )


def _load_eval_model(
    config: Mapping[str, Any],
    shape: tuple[int, int, int],
    stats: Mapping[str, Any] | None = None,
):
    specs = model_specs(config, shape)
    state_path = pretrained_path(config)
    if state_path is None:
        candidate = run_dir(config) / "model_state.pt"
        state_path = candidate if candidate.exists() else None
    LOGGER.debug("loading evaluation model state=%s", state_path)
    return (
        load_model(specs, state_dict_path=state_path, normalization_stats=stats)
        if state_path
        else load_model(specs, normalization_stats=stats)
    )


def evaluate_per_user_all(learner: TorchLearner, loader) -> dict[str, Any]:
    """Return elementwise losses grouped by individual id."""
    grouped: dict[str, dict[str, list[torch.Tensor]]] = {}
    names: dict[str, str] = {}
    learner.model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = learner._prepare_batch(raw_batch)
            prediction = learner._predict(batch)
            mean, std = get_normal_stats(batch.x, dim=-1, keepdim=True, detach=True)
            metadata = batch.metadata or {}
            ids = metadata.get("individual_ids")
            if ids is None:
                ids = torch.arange(batch.x.shape[0])
            ids = ids.detach().cpu().tolist() if torch.is_tensor(ids) else list(ids)
            batch_names = metadata.get("individual_names") or [str(value) for value in ids]
            for loss_name, criterion in learner.eval_losses.items():
                loss = criterion(
                    prediction,
                    batch.y,
                    context=batch.x,
                    mean=mean,
                    std=std,
                ).detach().cpu()
                grouped.setdefault(loss_name, {})
                for index, individual_id in enumerate(ids):
                    key = str(int(individual_id))
                    names[key] = str(batch_names[index])
                    grouped[loss_name].setdefault(key, []).append(loss[index])
    return {
        "losses": {
            loss_name: {
                individual_id: torch.stack(values, dim=0)
                for individual_id, values in per_loss.items()
            }
            for loss_name, per_loss in grouped.items()
        },
        "individual_names": names,
    }


def summarize_per_user(per_user: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return equal-user means and the worst-user 10% tail for each metric."""
    summary: dict[str, torch.Tensor] = {}
    for metric, users in (per_user.get("losses") or {}).items():
        user_means = torch.stack(
            [torch.as_tensor(values).float().mean() for values in users.values()]
        )
        tail = max(1, math.ceil(0.1 * user_means.numel()))
        summary[f"user_mean_{metric}"] = user_means.mean()
        summary[f"w10_{metric}"] = torch.topk(user_means, tail).values.mean()
    return summary


def plot_example_prediction(learner: TorchLearner, loader, *, save_path: str | Path | None = None):
    """Plot one prediction example; save only when ``save_path`` is provided."""
    import matplotlib.pyplot as plt

    raw_batch = next(iter(loader))
    batch = learner._prepare_batch(raw_batch)
    learner.model.eval()
    with torch.inference_mode():
        pred = learner._predict(batch).detach().cpu()
    x = batch.x.detach().cpu()[0, 0]
    y = batch.y.detach().cpu()[0, 0]
    p = pred[0, 0]
    lags = x.numel()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(lags), x, label="lookback")
    ax.plot(range(lags, lags + y.numel()), y, label="target")
    ax.plot(range(lags, lags + p.numel()), p, label="prediction")
    ax.axvline(lags - 1, color="black", linestyle=":")
    ax.legend(frameon=False)
    ax.set_title("Example forecast")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def eval_stage(
    config: Mapping[str, Any],
    *,
    model=None,
    learner: TorchLearner | None = None,
    loaders: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a model and save all-loss artifacts."""
    config = to_plain_config(config)
    stats = None
    if loaders is None:
        loaders, stats = _fetch_loaders(config)
    shape = tuple(get_sizes(loaders))
    training = section(config, "training")
    criterion, eval_losses = get_losses(
        training.get("loss", "nmse"),
        complete_evaluation=bool(training.get("complete_evaluation", True)),
    )
    if learner is None:
        if model is None:
            model = _load_eval_model(config, shape, stats)
        learner = TorchLearner(
            model,
            criterion,
            eval_losses=eval_losses,
            lr=float(training.get("lr", 1e-5)),
            device=device(config),
        )
    evaluation = section(config, "evaluation")
    selected = evaluation.get("splits")
    if isinstance(selected, str):
        selected = [selected]
    selected = selected or list(loaders)
    out_dir = run_dir(config)
    all_losses = {}
    per_user = {}
    for split in selected:
        if split not in loaders:
            LOGGER.warning("missing evaluation split=%s", split)
            continue
        all_losses[split] = learner.evaluate(
            loaders[split],
            return_mode="all",
            runs=int(evaluation.get("runs", 1)),
            seed=seed(config),
        )["losses"]
        per_user[split] = evaluate_per_user_all(learner, loaders[split])
        all_losses[split].update(summarize_per_user(per_user[split]))
    all_path = save_torch(all_losses, out_dir / "all_losses.pt")
    per_user_path = save_torch(per_user, out_dir / "per_user_all_losses.pt")
    example_path = None
    if bool(evaluation.get("plot_example", False)) and selected:
        example_path = Path(evaluation.get("example_plot_path", out_dir / "example_prediction.pdf"))
        plot_example_prediction(learner, loaders[selected[0]], save_path=example_path)
        LOGGER.info("saved example_prediction=%s", example_path.name)
    return {
        "all_losses": all_losses,
        "per_user_all_losses": per_user,
        "all_losses_path": Path(all_path),
        "per_user_all_losses_path": Path(per_user_path),
        "example_prediction_path": example_path,
        "learner": learner,
        "loaders": loaders,
    }


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup_logging(section(to_plain_config(config or {}), "misc").get("log_level", "INFO"))
    return eval_stage(config or {})
