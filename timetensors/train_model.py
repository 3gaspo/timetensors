"""Training stage for TimeTensor experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .dataset import fetch_training_data, get_sizes
from .load_dataset import build_dataset_stage
from .models import TorchLearner, cuda_diagnostics, get_losses, load_model
from .runtime import (
    batch_size,
    dataset_path,
    default_sampling,
    default_splits,
    default_subsets,
    device,
    model_specs,
    pretrained_path,
    rebuild_dataset,
    run_dir,
    save_torch,
    section,
    seed,
    setup_logging,
    task_shape,
    to_plain_config,
)


LOGGER = logging.getLogger(__name__)


def _cuda_summary() -> dict[str, Any]:
    return cuda_diagnostics()


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
    LOGGER.info("===== Running train script =====")
    LOGGER.info("Run directory: %s", run_dir(config))
    if loaders is None:
        if rebuild_dataset(config):
            LOGGER.info("Dataset rebuild requested before training")
            built = build_dataset_stage(config)
            loaders = built.get("loaders")
            stats = built.get("stats")
        else:
            LOGGER.info("Fetching existing tensor dataset from %s", dataset_path(config))
            loaders, stats = _fetch_loaders(config)
        LOGGER.info("Training data fetched")
    else:
        LOGGER.info("Using loaders already prepared by parent experiment")
    assert loaders is not None
    shape = tuple(get_sizes(loaders))
    LOGGER.info("Loader shape: %s", shape)
    try:
        _, split_info, batch_info = get_sizes(loaders, str_info=True)
        LOGGER.info("Loader splits:\n%s", split_info)
        LOGGER.info("Example batch:\n%s", batch_info)
    except Exception as exc:
        LOGGER.debug("Could not log detailed loader sizes: %s", exc)
    specs = model_specs(config, shape)
    init_path = pretrained_path(config)
    LOGGER.info("Building model: specs=%s pretrained=%s", specs, init_path)
    model = load_model(specs, state_dict_path=init_path) if init_path else load_model(specs)
    training = section(config, "training")
    criterion, eval_losses = get_losses(
        training.get("loss", "MSE"),
        complete_evaluation=bool(training.get("complete_evaluation", True)),
    )
    learner = TorchLearner(
        model,
        criterion,
        eval_losses=eval_losses,
        lr=float(training.get("lr", 1e-3)),
        device=device(config),
        optimizer_name=str(training.get("optimizer", "adam")),
        optimizer_kwargs=training.get("optimizer_kwargs"),
        grad_clip=training.get("grad_clip"),
    )
    LOGGER.info("Fetched model and learner")
    requested_device = device(config)
    LOGGER.info(
        "Device selected: requested=%s resolved=%s %s",
        requested_device,
        learner.device,
        _cuda_summary(),
    )
    epochs = int(training.get("epochs", 1))
    trainable = any(param.requires_grad for param in learner.model.parameters())
    if epochs > 0 and trainable:
        LOGGER.info("--Training--")
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
        LOGGER.info("Training stage finished")
    else:
        history = {
            "train": [],
            "valid": {},
            "elapsed_seconds": 0.0,
            "skipped": "no trainable parameters" if not trainable else "epochs <= 0",
        }
        LOGGER.info("Training skipped: %s", history["skipped"])
    out_dir = run_dir(config)
    state_path = learner.model.save_state_dict(out_dir / "model_state.pt")
    save_torch(history, out_dir / "train_history.pt")
    save_torch({"stats": stats, "shape": shape}, out_dir / "train_metadata.pt")
    LOGGER.info("Saved model state: %s", state_path)
    LOGGER.info("Saved training history: %s", out_dir / "train_history.pt")
    LOGGER.info("Saved training metadata: %s", out_dir / "train_metadata.pt")
    LOGGER.info("End of train script")
    return {
        "model": learner.model,
        "learner": learner,
        "history": history,
        "state_path": Path(state_path),
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
