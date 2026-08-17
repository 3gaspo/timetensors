"""Hydra entrypoint coordinating dataset build, training, and evaluation."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Mapping

from dataset.load import build_dataset_stage
from runtime import (
    config_bool,
    device,
    pretrained_path,
    rebuild_dataset,
    recompute_stats,
    run_dir,
    seed,
    seeded_configs,
    section,
    setup_logging,
    to_plain_config,
)
from training.evaluate import eval_stage
from training.per_user import train_per_user
from training.train import train_stage
from visu.experiment_plots import save_criterion_loss_plot


LOGGER = logging.getLogger(__name__)


def _log_device_once(
    learner: Any,
    *,
    requested: str,
    logged: bool,
) -> bool:
    if logged or learner is None:
        return logged
    resolved = getattr(learner, "device", "unknown")
    LOGGER.info(
        "device requested=%s resolved=%s cuda=%s",
        requested,
        resolved,
        str(resolved).startswith("cuda"),
    )
    return True


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the configured TimeTensor experiment."""
    config = to_plain_config(config)
    setup_logging(section(config, "misc").get("log_level", "INFO"))
    start = perf_counter()
    experiment = section(config, "experiment")
    training = section(config, "training")
    results: dict[str, Any] = {}
    loaders = None
    stats = None
    out_dir = run_dir(config)
    model_config = section(config, "model")
    model_name = model_config.get("name", model_config.get("path", "linear"))
    LOGGER.info("%s", "=" * 72)
    LOGGER.info(
        "experiment model=%s out=%s rebuild_dataset=%s evaluate=%s",
        model_name,
        out_dir,
        rebuild_dataset(config),
        bool(experiment.get("evaluate", True)),
    )
    LOGGER.info("stats recompute=%s", recompute_stats(config))
    logged_device = False

    if rebuild_dataset(config):
        LOGGER.info("dataset start")
        dataset_result = build_dataset_stage(config)
        loaders = dataset_result.get("loaders")
        stats = dataset_result.get("stats")
        results["dataset_path"] = str(dataset_result["dataset_path"])
        results["shape"] = dataset_result.get("shape")
        LOGGER.info("dataset done")
    else:
        LOGGER.info("dataset rebuild skipped")

    has_pretrained = pretrained_path(config) is not None
    skip_training = bool(experiment.get("skip_training", False)) or (
        has_pretrained and bool(experiment.get("bypass_training_with_pretrained", True))
    )
    train_result = None
    if not skip_training:
        LOGGER.info("training start")
        train_result = train_stage(config, loaders=loaders, stats=stats)
        loaders = train_result["loaders"]
        stats = train_result["stats"]
        results["state_path"] = str(train_result["state_path"])
        results["train_history_path"] = str(out_dir / "train_history.pt")
        if train_result.get("weight_plot_paths"):
            results["linear_weight_plot_path"] = str(train_result["weight_plot_paths"]["image"])
            results["linear_weight_tensor_path"] = str(train_result["weight_plot_paths"]["tensor"])
        logged_device = True
        history = train_result["history"]
        if history.get("train"):
            criterion_name = train_result["learner"].criterion.name
            plot_path = save_criterion_loss_plot(
                history,
                criterion_name,
                out_dir / "criterion_loss.pdf",
                plot_step_train_loss=config_bool(training.get("plot_step_train_loss", True)),
            )
            results["criterion_loss_plot_path"] = str(plot_path)
            LOGGER.info("saved criterion_loss=%s", plot_path.name)
            LOGGER.info("training done")
        else:
            LOGGER.info("no training required reason=%s", history.get("skipped", "unknown"))
    elif has_pretrained:
        results["state_path"] = str(pretrained_path(config))
        LOGGER.info("training skipped pretrained=true")
    else:
        candidate = out_dir / "model_state.pt"
        if candidate.exists():
            results["state_path"] = str(candidate)
            LOGGER.info("training skipped existing_state=true")
        else:
            LOGGER.info("training skipped existing_state=false")

    if bool(experiment.get("evaluate", True)):
        LOGGER.info("evaluation start")
        eval_result = eval_stage(
            config,
            model=None if train_result is None else train_result["model"],
            learner=None if train_result is None else train_result["learner"],
            loaders=loaders,
        )
        results["all_losses_path"] = str(eval_result["all_losses_path"])
        if eval_result.get("example_prediction_path") is not None:
            results["example_prediction_path"] = str(eval_result["example_prediction_path"])
        logged_device = _log_device_once(
            eval_result.get("learner"),
            requested=device(config),
            logged=logged_device,
        )
        LOGGER.info("evaluation done")
    else:
        LOGGER.info("evaluation skipped")

    LOGGER.info("experiment done seconds=%.2f", perf_counter() - start)
    LOGGER.info("%s", "=" * 72)
    return results


def _run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    scope = str(section(config, "experiment").get("training_scope", "central"))
    if scope == "per_user":
        return train_per_user(config)
    if scope != "central":
        raise ValueError(f"unknown experiment.training_scope {scope!r}")
    return run_experiment(config)


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    configs = seeded_configs(config or {})
    if len(configs) == 1:
        return _run_config(configs[0])
    return {int(seed(item)): _run_config(item) for item in configs}


try:
    import hydra  # type: ignore
except Exception:  # pragma: no cover
    hydra = None


if hydra is not None:

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def _hydra_main(cfg):
        main(cfg)


if __name__ == "__main__":
    if hydra is None:
        main({})
    else:
        _hydra_main()
