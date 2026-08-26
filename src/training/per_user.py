"""Train one independent forecasting model per training user."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from pipeline.runtime import run_dir, save_name, save_torch, section, to_plain_config
from visualization.experiment_plots import save_criterion_loss_plot

from .evaluate import eval_stage, merge_loss_payloads
from .train import fetch_loaders, train_stage


def _restrict_to_user(loaders: Mapping[str, Any], user_id: int) -> dict[str, Any]:
    selected = {}
    for split, loader in loaders.items():
        dataset = loader.dataset
        if not hasattr(dataset, "data"):
            continue
        ids = [int(value) for value in dataset.data.individual_ids.tolist()]
        if user_id not in ids:
            continue
        dataset.set_sampler(subset_mode="individuals", subset_indices=[ids.index(user_id)])
        selected[split] = loader
    return selected


def train_per_user(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit/evaluate a fresh model for every user visible in the train split."""
    config = to_plain_config(config)
    base_name = save_name(config)
    initial_loaders, _ = fetch_loaders(config)
    user_ids = [int(value) for value in initial_loaders["train"].dataset.data.individual_ids]
    split_payloads: dict[str, list[Mapping[str, Any]]] = {}

    for user_id in user_ids:
        user_config = deepcopy(config)
        user_config.setdefault("output", {})["name"] = f"{base_name}/users/user_{user_id}"
        loaders, stats = fetch_loaders(user_config)
        loaders = _restrict_to_user(loaders, user_id)
        trained = train_stage(user_config, loaders=loaders, stats=stats)
        evaluation = eval_stage(
            user_config,
            model=trained["model"],
            learner=trained["learner"],
            loaders=loaders,
        )
        save_criterion_loss_plot(
            trained["history"],
            section(user_config, "training").get("loss", "nmse"),
            run_dir(user_config) / "criterion_loss.pdf",
            plot_step_train_loss=bool(
                section(user_config, "training").get("plot_step_train_loss", False)
            ),
        )
        for split, payload in evaluation["all_losses"].items():
            split_payloads.setdefault(split, []).append(payload)

    all_losses = {
        split: merge_loss_payloads(payloads)
        for split, payloads in split_payloads.items()
    }

    output = run_dir(config)
    all_path = save_torch(all_losses, output / "all_losses.pt")
    return {
        "all_losses": all_losses,
        "all_losses_path": all_path,
    }
