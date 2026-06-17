"""Interactive plotting helpers for TimeTensor experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def symlog(x, linthresh: float = 1.0):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x / linthresh)) * linthresh


def load_pt(path: str | Path):
    path = Path(path)
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def load_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def plot_losses(train_losses, valid=None, title: str = "Training losses", logscale: bool = True):
    fig, ax = plt.subplots(figsize=(10, 4))
    if train_losses is not None and len(train_losses):
        ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train")
    for name, values in (valid or {}).items():
        if isinstance(values, list) and values and isinstance(values[0], dict):
            keys = sorted(values[0]["losses"]) if "losses" in values[0] else []
            steps = [value.get("step", index + 1) for index, value in enumerate(values)]
            for key in keys:
                losses = [float(value["losses"][key]) for value in values]
                ax.plot(steps, losses, marker="o", label=f"{name}:{key}")
        else:
            ax.plot(values, label=name)
    if logscale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    return fig


def plot_error_distribution(loss_tensor, title: str = "Loss distribution"):
    values = loss_tensor.float().mean(dim=tuple(range(1, loss_tensor.ndim))).numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(values, log_scale=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("sample loss")
    return fig


def plot_horizon_errors(loss_tensor, title: str = "Mean error by horizon"):
    values = loss_tensor.float().mean(
        dim=tuple(index for index in range(loss_tensor.ndim) if index != loss_tensor.ndim - 1)
    ).numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(values)), values)
    ax.set_title(title)
    ax.set_xlabel("horizon")
    ax.set_ylabel("mean loss")
    return fig


def plot_per_user_scatter(per_user_loss, loss_name: str, split: str):
    rows = []
    names = per_user_loss.get("individual_names", {})
    for user_id, tensor in per_user_loss["losses"][loss_name].items():
        sample = tensor.float().mean(dim=tuple(range(1, tensor.ndim))).numpy()
        rows.append(
            {
                "user_id": user_id,
                "name": names.get(user_id, user_id),
                "mean": sample.mean(),
                "std": sample.std(),
            }
        )
    frame = pd.DataFrame(rows)
    frame["symlog_mean"] = symlog(frame["mean"])
    frame["symlog_std"] = symlog(frame["std"])
    grid = sns.jointplot(data=frame, x="symlog_mean", y="symlog_std", kind="scatter")
    grid.figure.suptitle(f"Per-user {loss_name} on {split}", y=1.02)
    return grid.figure, frame


def plot_boxplot(per_user_loss, loss_name: str, split: str):
    rows = []
    for user_id, tensor in per_user_loss["losses"][loss_name].items():
        values = tensor.float().mean(dim=tuple(range(1, tensor.ndim))).numpy()
        rows.extend({"user_id": user_id, "loss": value} for value in values)
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=frame, x="user_id", y="loss", ax=ax)
    ax.set_title(f"Per-user {loss_name} boxplots on {split}")
    ax.tick_params(axis="x", rotation=90)
    return fig


def plot_heatmap(matrix, title: str = "Heatmap", x_name: str = "x", y_name: str = "y"):
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(np.asarray(matrix), aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    return fig


def results_table(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    json_files = sorted(run_dir.glob("*results.json")) + sorted(run_dir.glob("*summary.json"))
    return {path.name: load_json(path) for path in json_files}


def display_dashboard(default_run_dir: str | Path = "../outputs/model") -> None:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    run_dir_widget = widgets.Text(
        value=str(default_run_dir),
        description="run_dir",
        layout=widgets.Layout(width="700px"),
    )
    reload_button = widgets.Button(description="Load artifacts", button_style="primary")
    plot_kind = widgets.Dropdown(
        options=[
            "training",
            "loss_distribution",
            "horizon_errors",
            "per_user_scatter",
            "per_user_boxplot",
            "results_json",
        ],
        description="plot",
    )
    split_widget = widgets.Dropdown(options=[], description="split")
    loss_widget = widgets.Dropdown(options=[], description="loss")
    logscale_widget = widgets.Checkbox(value=True, description="log training")
    output = widgets.Output()
    state: dict[str, Any] = {}

    def refresh_options():
        all_losses = state.get("all_losses") or {}
        splits = list(all_losses) or list((state.get("per_user") or {}))
        split_widget.options = splits
        if splits:
            split_widget.value = splits[0]
            losses = list(all_losses.get(splits[0], {}))
            if not losses and state.get("per_user", {}).get(splits[0]):
                losses = list(state["per_user"][splits[0]]["losses"])
            loss_widget.options = losses
            if losses:
                loss_widget.value = losses[0]

    def load_artifacts(_=None):
        run_dir = Path(run_dir_widget.value)
        state.clear()
        state["run_dir"] = run_dir
        state["history"] = load_pt(run_dir / "train_history.pt")
        state["all_losses"] = load_pt(run_dir / "all_losses.pt") or {}
        state["per_user"] = load_pt(run_dir / "per_user_all_losses.pt") or {}
        state["results"] = results_table(run_dir)
        refresh_options()
        with output:
            clear_output()
            print("Loaded from", run_dir)
            print("history:", state["history"] is not None)
            print("splits:", list(state["all_losses"]) or list(state["per_user"]))

    def draw(_=None):
        with output:
            clear_output()
            kind = plot_kind.value
            split = split_widget.value
            loss_name = loss_widget.value
            if kind == "training":
                history = state.get("history") or {}
                display(plot_losses(history.get("train"), history.get("valid"), logscale=logscale_widget.value))
            elif kind == "results_json":
                display(state.get("results", {}))
            elif split and loss_name and kind == "loss_distribution":
                display(plot_error_distribution(state["all_losses"][split][loss_name], f"{split} {loss_name} distribution"))
            elif split and loss_name and kind == "horizon_errors":
                display(plot_horizon_errors(state["all_losses"][split][loss_name], f"{split} {loss_name} by horizon"))
            elif split and loss_name and kind == "per_user_scatter":
                fig, frame = plot_per_user_scatter(state["per_user"][split], loss_name, split)
                display(fig)
                display(frame.sort_values("mean", ascending=False).head(20))
            elif split and loss_name and kind == "per_user_boxplot":
                display(plot_boxplot(state["per_user"][split], loss_name, split))

    reload_button.on_click(load_artifacts)
    for widget in [plot_kind, split_widget, loss_widget, logscale_widget]:
        widget.observe(draw, names="value")

    display(
        widgets.VBox(
            [
                widgets.HBox([run_dir_widget, reload_button]),
                widgets.HBox([plot_kind, split_widget, loss_widget, logscale_widget]),
                output,
            ]
        )
    )
    load_artifacts()
