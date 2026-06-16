"""Hydra entrypoint coordinating dataset build, training, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .eval_model import eval_stage
from .load_dataset import build_dataset_stage
from .runtime import pretrained_path, rebuild_dataset, run_dir, save_json, section, setup_logging, to_plain_config
from .train_model import train_stage


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the configured TimeTensor experiment."""
    config = to_plain_config(config)
    setup_logging(section(config, "misc").get("log_level", "INFO"))
    experiment = section(config, "experiment")
    results: dict[str, Any] = {}
    loaders = None
    stats = None

    if rebuild_dataset(config):
        dataset_result = build_dataset_stage(config)
        loaders = dataset_result.get("loaders")
        stats = dataset_result.get("stats")
        results["dataset_path"] = str(dataset_result["dataset_path"])
        results["shape"] = dataset_result.get("shape")

    has_pretrained = pretrained_path(config) is not None
    skip_training = bool(experiment.get("skip_training", False)) or (
        has_pretrained and bool(experiment.get("bypass_training_with_pretrained", True))
    )
    train_result = None
    if not skip_training:
        train_result = train_stage(config, loaders=loaders, stats=stats)
        loaders = train_result["loaders"]
        stats = train_result["stats"]
        results["state_path"] = str(train_result["state_path"])
        results["train_history_path"] = str(run_dir(config) / "train_history.pt")
    elif has_pretrained:
        results["state_path"] = str(pretrained_path(config))
    else:
        candidate = run_dir(config) / "model_state.pt"
        if candidate.exists():
            results["state_path"] = str(candidate)

    if bool(experiment.get("evaluate", True)):
        eval_result = eval_stage(
            config,
            model=None if train_result is None else train_result["model"],
            learner=None if train_result is None else train_result["learner"],
            loaders=loaders,
        )
        results["all_losses_path"] = str(eval_result["all_losses_path"])
        results["per_user_all_losses_path"] = str(eval_result["per_user_all_losses_path"])

    summary_path = save_json(results, run_dir(config) / "experiment_summary.json")
    results["summary_path"] = str(Path(summary_path))
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
