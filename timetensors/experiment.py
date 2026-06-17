"""Hydra entrypoint coordinating dataset build, training, and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .eval_model import eval_stage
from .load_dataset import build_dataset_stage
from .runtime import pretrained_path, rebuild_dataset, run_dir, save_json, section, setup_logging, to_plain_config
from .train_model import train_stage


LOGGER = logging.getLogger(__name__)


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the configured TimeTensor experiment."""
    config = to_plain_config(config)
    setup_logging(section(config, "misc").get("log_level", "INFO"))
    LOGGER.info("===== Running experiment script =====")
    experiment = section(config, "experiment")
    results: dict[str, Any] = {}
    loaders = None
    stats = None
    out_dir = run_dir(config)
    LOGGER.info(
        "Experiment configuration: run_dir=%s rebuild_dataset=%s evaluate=%s",
        out_dir,
        rebuild_dataset(config),
        bool(experiment.get("evaluate", True)),
    )

    if rebuild_dataset(config):
        LOGGER.info("Launching dataset stage")
        dataset_result = build_dataset_stage(config)
        loaders = dataset_result.get("loaders")
        stats = dataset_result.get("stats")
        results["dataset_path"] = str(dataset_result["dataset_path"])
        results["shape"] = dataset_result.get("shape")
        LOGGER.info("Dataset stage finished: dataset_path=%s shape=%s", results["dataset_path"], results.get("shape"))
    else:
        LOGGER.info("Dataset rebuild skipped by config")

    has_pretrained = pretrained_path(config) is not None
    skip_training = bool(experiment.get("skip_training", False)) or (
        has_pretrained and bool(experiment.get("bypass_training_with_pretrained", True))
    )
    train_result = None
    if not skip_training:
        LOGGER.info("Launching training stage")
        train_result = train_stage(config, loaders=loaders, stats=stats)
        loaders = train_result["loaders"]
        stats = train_result["stats"]
        results["state_path"] = str(train_result["state_path"])
        results["train_history_path"] = str(out_dir / "train_history.pt")
        LOGGER.info("Training stage finished: state_path=%s", results["state_path"])
    elif has_pretrained:
        results["state_path"] = str(pretrained_path(config))
        LOGGER.info("Training skipped: using pretrained state %s", results["state_path"])
    else:
        candidate = out_dir / "model_state.pt"
        if candidate.exists():
            results["state_path"] = str(candidate)
            LOGGER.info("Training skipped: using existing state %s", results["state_path"])
        else:
            LOGGER.info("Training skipped and no existing state was found")

    if bool(experiment.get("evaluate", True)):
        LOGGER.info("Launching evaluation stage")
        eval_result = eval_stage(
            config,
            model=None if train_result is None else train_result["model"],
            learner=None if train_result is None else train_result["learner"],
            loaders=loaders,
        )
        results["all_losses_path"] = str(eval_result["all_losses_path"])
        results["per_user_all_losses_path"] = str(eval_result["per_user_all_losses_path"])
        LOGGER.info(
            "Evaluation stage finished: all_losses=%s per_user_all_losses=%s",
            results["all_losses_path"],
            results["per_user_all_losses_path"],
        )
    else:
        LOGGER.info("Evaluation skipped by config")

    summary_path = save_json(results, out_dir / "experiment_summary.json")
    results["summary_path"] = str(Path(summary_path))
    LOGGER.info("Saved experiment summary: %s", results["summary_path"])
    LOGGER.info("End of experiment script")
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
