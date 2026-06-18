"""Training stage for TimeTensor experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .dataset import fetch_training_data, get_sizes
from .load_dataset import build_dataset_stage
from .models import TorchLearner, get_losses, load_model
from .runtime import (
    batch_size,
    config_bool,
    dataset_path,
    default_sampling,
    default_splits,
    default_subsets,
    device,
    model_specs,
    pretrained_path,
    recompute_stats,
    rebuild_dataset,
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


def _log_loader_sizes(loaders: Mapping[str, Any]) -> None:
    try:
        shape, split_info, batch_info = get_sizes(loaders, str_info=True)
        LOGGER.info("data_shape=%s", shape)
        LOGGER.info("%s", split_info)
        LOGGER.info("%s", batch_info)
    except Exception as exc:
        LOGGER.debug("could not log loader sizes: %s", exc)


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


def train_stage(
    config: Mapping[str, Any],
    *,
    loaders: Mapping[str, Any] | None = None,
    stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a model and save state/history artifacts."""
    config = to_plain_config(config)
    if loaders is None:
        if rebuild_dataset(config):
            built = build_dataset_stage(config)
            loaders = built.get("loaders")
            stats = built.get("stats")
        else:
            loaders, stats = _fetch_loaders(config)
    assert loaders is not None
    shape = tuple(get_sizes(loaders))
    specs = model_specs(config, shape)
    init_path = pretrained_path(config)
    model = (
        load_model(specs, state_dict_path=init_path, normalization_stats=stats)
        if init_path
        else load_model(specs, normalization_stats=stats)
    )
    training = section(config, "training")
    criterion, eval_losses = get_losses(
        training.get("loss", "nmse"),
        complete_evaluation=bool(training.get("complete_evaluation", True)),
    )
    learner = TorchLearner(
        model,
        criterion,
        eval_losses=eval_losses,
        lr=float(training.get("lr", 1e-5)),
        device=device(config),
        optimizer_name=str(training.get("optimizer", "adam")),
        optimizer_kwargs=training.get("optimizer_kwargs"),
        grad_clip=training.get("grad_clip"),
    )
    requested_device = device(config)
    LOGGER.info(
        "device requested=%s resolved=%s cuda=%s",
        requested_device,
        learner.device,
        str(learner.device).startswith("cuda"),
    )
    _log_loader_sizes(loaders)
    epochs = int(training.get("epochs", 200))
    trainable = any(param.requires_grad for param in learner.model.parameters())
    if epochs > 0 and trainable:
        history = learner.fit(
            loaders["train"],
            epochs=epochs,
            valid_loaders={key: value for key, value in loaders.items() if "valid" in key},
            eval_every_steps=training.get("eval_every_steps", 100),
            log_every_steps=training.get("log_every_steps", 1000),
            eval_runs=int(training.get("eval_runs", 1)),
            seed=seed(config),
            logger=LOGGER,
        )
    else:
        history = {
            "train": [],
            "valid": {},
            "elapsed_seconds": 0.0,
            "skipped": "no trainable parameters" if not trainable else "epochs <= 0",
        }
    out_dir = run_dir(config)
    state_path = learner.model.save_state_dict(out_dir / "model_state.pt")
    save_torch(history, out_dir / "train_history.pt")
    save_torch({"stats": stats, "shape": shape}, out_dir / "train_metadata.pt")
    weight_plot_paths = None
    if config_bool(section(config, "experiment").get("plot_weights", False)):
        try:
            weight_plot_paths = save_linear_weight_plots(
                learner.model,
                out_dir,
                title=f"{section(config, 'model').get('name', 'model')} weights",
            )
            LOGGER.info("saved linear_weights=%s", weight_plot_paths["image"].name)
        except ValueError as exc:
            LOGGER.debug("could not plot model weights: %s", exc)
    LOGGER.debug("saved training artifacts in %s", out_dir)
    return {
        "model": learner.model,
        "learner": learner,
        "history": history,
        "state_path": Path(state_path),
        "weight_plot_paths": weight_plot_paths,
        "loaders": loaders,
        "stats": stats,
        "shape": shape,
    }


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup_logging(section(to_plain_config(config or {}), "misc").get("log_level", "INFO"))
    return train_stage(config or {})


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
