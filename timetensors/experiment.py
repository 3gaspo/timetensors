"""Hydra entrypoint coordinating dataset build, training, and evaluation."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Mapping

from .dataset import get_sizes
from .eval_model import eval_stage
from .load_dataset import build_dataset_stage
from .runtime import pretrained_path, rebuild_dataset, recompute_stats, run_dir, section, setup_logging, to_plain_config
from .train_model import train_stage
from .visu.experiment_plots import save_criterion_loss_plot


LOGGER = logging.getLogger(__name__)


def _log_loader_sizes(loaders: Mapping[str, Any] | None) -> None:
    if not loaders:
        return
    try:
        shape, split_info, batch_info = get_sizes(loaders, str_info=True)
        LOGGER.info("data_shape=%s", shape)
        LOGGER.info("%s", split_info)
        LOGGER.info("%s", batch_info)
    except Exception as exc:
        LOGGER.debug("could not log loader sizes: %s", exc)


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the configured TimeTensor experiment."""
    config = to_plain_config(config)
    setup_logging(section(config, "misc").get("log_level", "INFO"))
    start = perf_counter()
    experiment = section(config, "experiment")
    results: dict[str, Any] = {}
    loaders = None
    stats = None
    out_dir = run_dir(config)
    LOGGER.info(
        "experiment out=%s rebuild_dataset=%s evaluate=%s",
        out_dir,
        rebuild_dataset(config),
        bool(experiment.get("evaluate", True)),
    )
    LOGGER.info("stats recompute=%s", recompute_stats(config))
    logged_sizes = False

    if rebuild_dataset(config):
        LOGGER.info("dataset start")
        dataset_result = build_dataset_stage(config)
        loaders = dataset_result.get("loaders")
        stats = dataset_result.get("stats")
        results["dataset_path"] = str(dataset_result["dataset_path"])
        results["shape"] = dataset_result.get("shape")
        LOGGER.info("dataset done")
        _log_loader_sizes(loaders)
        logged_sizes = loaders is not None
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
        if not logged_sizes:
            _log_loader_sizes(loaders)
            logged_sizes = True
        history = train_result["history"]
        if history.get("train"):
            criterion_name = train_result["learner"].criterion.name
            plot_path = save_criterion_loss_plot(
                history,
                criterion_name,
                out_dir / "criterion_loss.png",
            )
            results["criterion_loss_plot_path"] = str(plot_path)
            LOGGER.info("saved criterion_loss=%s", plot_path.name)
        LOGGER.info("training done")
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
        results["per_user_all_losses_path"] = str(eval_result["per_user_all_losses_path"])
        if eval_result.get("example_prediction_path") is not None:
            results["example_prediction_path"] = str(eval_result["example_prediction_path"])
        LOGGER.info("evaluation done")
    else:
        LOGGER.info("evaluation skipped")

    LOGGER.info("experiment done seconds=%.2f", perf_counter() - start)
    return results


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return run_experiment(config or {})


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
