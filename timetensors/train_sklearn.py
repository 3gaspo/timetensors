"""Fit and evaluate scikit-learn forecasting models on TimeTensor loaders."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch

from .dataset import fetch_training_data, get_sizes
from .models import SkLinearForecaster, get_losses, iter_loader_xy
from .models.normalizations import get_normal_stats
from .runtime import (
    batch_size,
    config_bool,
    dataset_path,
    default_sampling,
    default_splits,
    default_subsets,
    recompute_stats,
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
from .visu.experiment_plots import save_linear_weight_plots


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
        stats_save_path=run_dir(config) / "dataset_artifacts",
        compute_stats=recompute_stats(config),
        stats_max_windows=stats_max_windows(config),
        stats_seed=stats_seed(config),
        stats_eps=stats_eps(config),
        legacy_context_kind=data_cfg.get("legacy_context_kind"),
    )


def _normalization_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    normalization = section(config, "normalization")
    if normalization.get("name") is None and not normalization.get("kwargs"):
        return None
    return {
        "name": normalization.get("name", "identity"),
        "kwargs": normalization.get("kwargs", {}) or {},
    }


def predict_loader(
    model: SkLinearForecaster,
    loader,
    *,
    unroll_mode: str = "accessible",
    max_windows: int | None = None,
):
    """Yield raw inputs, targets, and SkLinear predictions for a loader."""
    for x, y in iter_loader_xy(loader, mode=unroll_mode, max_windows=max_windows):
        yield x, y, model.predict(x)


def evaluate_sklearn(
    model: SkLinearForecaster,
    loader,
    eval_losses: Mapping[str, Any],
    *,
    unroll_mode: str = "accessible",
    max_windows: int | None = None,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in eval_losses}
    with torch.inference_mode():
        for x, y, pred in predict_loader(
            model,
            loader,
            unroll_mode=unroll_mode,
            max_windows=max_windows,
        ):
            mean, std = get_normal_stats(x, dim=-1, keepdim=True, detach=True)
            for name, criterion in eval_losses.items():
                collected[name].append(
                    criterion(pred, y, context=x, mean=mean, std=std).detach().cpu()
                )
    return {
        name: torch.cat(values, dim=0) if values else torch.empty(0)
        for name, values in collected.items()
    }


def train_sklearn_stage(
    config: Mapping[str, Any],
    *,
    loaders: Mapping[str, Any] | None = None,
    stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a scikit-learn model and save prediction/loss artifacts."""
    config = to_plain_config(config)
    start = perf_counter()
    if loaders is None:
        loaders, stats = _fetch_loaders(config)
    assert loaders is not None
    shape = tuple(get_sizes(loaders))
    lags, dim, horizon = shape
    model_cfg = section(config, "model")
    model_name = str(model_cfg.get("name", model_cfg.get("path", "sklinear"))).lower()
    if model_name != "sklinear":
        raise ValueError("train_sklearn currently supports model.name=sklinear only")

    sklearn_cfg = section(config, "sklearn")
    unroll_mode = str(sklearn_cfg.get("unroll_mode", "accessible"))
    eval_unroll_mode = str(sklearn_cfg.get("eval_unroll_mode", unroll_mode))
    max_windows = sklearn_cfg.get("max_windows")
    max_windows = None if max_windows in {None, "None", "none", ""} else int(max_windows)
    eval_max_windows = sklearn_cfg.get("eval_max_windows")
    eval_max_windows = None if eval_max_windows in {None, "None", "none", ""} else int(eval_max_windows)

    model_kwargs = dict(model_cfg.get("kwargs") or {})
    model_kwargs.update(dict(sklearn_cfg.get("model_kwargs") or {}))
    model = SkLinearForecaster(
        lags,
        dim,
        horizon,
        normalization=_normalization_config(config),
        normalization_stats=stats,
        model_kwargs=model_kwargs,
    )
    fit_info = model.fit_loader(
        loaders["train"],
        unroll_mode=unroll_mode,
        max_windows=max_windows,
    )
    fit_info["elapsed_seconds"] = perf_counter() - start
    LOGGER.info(
        "sklinear fit windows=%s features=%s targets=%s seconds=%.2f",
        fit_info["windows"],
        fit_info["features"],
        fit_info["targets"],
        fit_info["elapsed_seconds"],
    )

    training = section(config, "training")
    _, eval_losses = get_losses(
        training.get("loss", "nmse"),
        complete_evaluation=bool(training.get("complete_evaluation", True)),
    )
    evaluation = section(config, "evaluation")
    selected = evaluation.get("splits")
    if isinstance(selected, str):
        selected = [selected]
    selected = selected or list(loaders)

    all_losses = {}
    for split in selected:
        if split not in loaders:
            LOGGER.warning("missing evaluation split=%s", split)
            continue
        all_losses[split] = evaluate_sklearn(
            model,
            loaders[split],
            eval_losses,
            unroll_mode=eval_unroll_mode,
            max_windows=eval_max_windows,
        )

    out_dir = run_dir(config)
    model_path = model.save(out_dir / "sklinear_model.pt")
    metadata_path = save_torch({"stats": stats, "shape": shape, "fit": fit_info}, out_dir / "train_metadata.pt")
    history_path = save_torch({"fit": fit_info, "train": [], "valid": {}}, out_dir / "train_history.pt")
    all_losses_path = save_torch(all_losses, out_dir / "all_losses.pt")
    weight_plot_paths = None
    if config_bool(section(config, "experiment").get("plot_weights", True)):
        weight_plot_paths = save_linear_weight_plots(
            model,
            out_dir,
            title=f"{model_name} weights",
        )
        LOGGER.info("saved linear_weights=%s", weight_plot_paths["image"].name)

    return {
        "model": model,
        "loaders": loaders,
        "stats": stats,
        "fit": fit_info,
        "state_path": Path(model_path),
        "train_metadata_path": Path(metadata_path),
        "train_history_path": Path(history_path),
        "all_losses": all_losses,
        "all_losses_path": Path(all_losses_path),
        "weight_plot_paths": weight_plot_paths,
    }


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup_logging(section(to_plain_config(config or {}), "misc").get("log_level", "INFO"))
    return train_sklearn_stage(config or {})


try:
    import hydra  # type: ignore
except Exception:  # pragma: no cover
    hydra = None


if hydra is not None:

    @hydra.main(version_base=None, config_path=None, config_name=None)
    def _hydra_main(cfg):
        main(cfg)


if __name__ == "__main__":
    if hydra is None:
        main({})
    else:
        _hydra_main()
