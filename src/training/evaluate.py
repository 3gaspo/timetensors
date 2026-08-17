"""Evaluation stage for TimeTensor experiments."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from dataset import fetch_training_data, get_sizes
from models import load_model
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


def build_loss_payload(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact, row-aligned artifact for one evaluation split."""
    losses = dict(evaluation.get("losses") or {})
    metadata = dict(evaluation.get("metadata") or {})
    lengths = {
        int(value.shape[0])
        for value in losses.values()
        if torch.is_tensor(value) and value.ndim > 0
    }
    if len(lengths) > 1:
        raise ValueError(f"loss tensors have inconsistent sample counts: {sorted(lengths)}")
    sample_count = next(iter(lengths), 0)
    for key in ("individual_ids", "query_ids", "run_ids"):
        value = metadata.get(key)
        if value is not None and torch.as_tensor(value).numel() != sample_count:
            raise ValueError(f"metadata {key!r} is not aligned with {sample_count} losses")
    if sample_count and metadata.get("individual_ids") is None:
        raise ValueError("complete TimeTensors evaluation requires stable individual_ids")
    payload = {"losses": losses, "metadata": metadata, "summaries": {}}
    payload["summaries"] = summarize_per_user(payload)
    return payload


def summarize_per_user(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Aggregate aligned elementwise losses into equal-user and W10 metrics."""
    ids = (payload.get("metadata") or {}).get("individual_ids")
    if ids is None:
        return {}
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        return {}
    summary: dict[str, torch.Tensor] = {}
    for metric, values in (payload.get("losses") or {}).items():
        values = torch.as_tensor(values).float()
        if values.shape[0] != ids.numel():
            raise ValueError(f"loss {metric!r} is not aligned with individual_ids")
        user_means = torch.stack(
            [values[ids == individual_id].mean() for individual_id in torch.unique(ids, sorted=True)]
        )
        tail = max(1, math.ceil(0.1 * user_means.numel()))
        summary[f"user_mean_{metric}"] = user_means.mean()
        summary[f"w10_{metric}"] = torch.topk(user_means, tail).values.mean()
    return summary


def merge_loss_payloads(
    payloads: list[Mapping[str, Any]],
    *,
    report_equal_user_metrics: bool = False,
) -> dict[str, Any]:
    """Concatenate aligned split payloads without materializing per-user copies."""
    if not payloads:
        return {"losses": {}, "metadata": {}, "summaries": {}}
    metric_names = set(payloads[0].get("losses") or {})
    if any(set(payload.get("losses") or {}) != metric_names for payload in payloads[1:]):
        raise ValueError("cannot merge evaluation payloads with different loss metrics")
    losses = {
        metric: torch.cat([torch.as_tensor(payload["losses"][metric]) for payload in payloads])
        for metric in metric_names
    }
    metadata: dict[str, Any] = {}
    for key in ("individual_ids", "query_ids", "run_ids"):
        values = [(payload.get("metadata") or {}).get(key) for payload in payloads]
        if all(value is not None for value in values):
            metadata[key] = torch.cat([torch.as_tensor(value).reshape(-1) for value in values])
        elif any(value is not None for value in values):
            raise ValueError(f"metadata {key!r} must be present in every merged payload")
    names: dict[str, str] = {}
    for payload in payloads:
        names.update((payload.get("metadata") or {}).get("individual_names") or {})
    if names:
        metadata["individual_names"] = names
    merged = {"losses": losses, "metadata": metadata, "summaries": {}}
    summaries = summarize_per_user(merged)
    if report_equal_user_metrics:
        summaries.update(
            {
                metric: summaries[f"user_mean_{metric}"]
                for metric in metric_names
            }
        )
    merged["summaries"] = summaries
    return merged


def plot_example_prediction(example: Mapping[str, torch.Tensor], *, save_path: str | Path | None = None):
    """Plot an example captured during evaluation without another inference."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = example["inputs"][0, 0]
    y = example["targets"][0, 0]
    p = example["predictions"][0, 0]
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
    example = None
    plot_example = bool(evaluation.get("plot_example", False))
    for split in selected:
        if split not in loaders:
            LOGGER.warning("missing evaluation split=%s", split)
            continue
        result = learner.evaluate(
            loaders[split],
            return_mode="all",
            runs=int(evaluation.get("runs", 1)),
            seed=seed(config),
            capture_example=plot_example and example is None,
        )
        all_losses[split] = build_loss_payload(result)
        if example is None:
            example = result.get("example")
    all_path = save_torch(all_losses, out_dir / "all_losses.pt")
    example_path = None
    if plot_example and example is not None:
        example_path = Path(evaluation.get("example_plot_path", out_dir / "example_prediction.pdf"))
        plot_example_prediction(example, save_path=example_path)
        LOGGER.info("saved example_prediction=%s", example_path.name)
    return {
        "all_losses": all_losses,
        "all_losses_path": Path(all_path),
        "example_prediction_path": example_path,
        "learner": learner,
        "loaders": loaders,
    }


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup_logging(section(to_plain_config(config or {}), "misc").get("log_level", "INFO"))
    return eval_stage(config or {})
