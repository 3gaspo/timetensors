"""Run a dependency-light smoke benchmark on the archived synthetic population.

This intentionally uses NumPy rather than the main PyTorch trainer so that the
complete experiment matrix can be checked on a laptop without a prepared uv
environment.  It exercises the same data policies, losses, normalizations,
sampling modes, metrics, seed aggregation, and table formatting as the Slurm
benchmarks, using a linear multi-output forecaster for six epochs.
"""

from __future__ import annotations

import copy
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from pipeline.runs import (
    SelectedRun,
    allocate_run,
    identity_path,
    load_manifest,
    mark_status,
    write_report_manifest,
)


SEEDS = (1, 2)
LAGS = 24
HORIZON = 6
EPOCHS = 6
BENCHMARK_FAMILIES = (
    "constants",
    "sampling",
    "normalizations",
    "reference",
    "losses",
    "linear_models",
    "central_per_user",
)
EXPECTED_METHODS = {
    "constants": {"keep", "remove_train_windows", "remove_eval_windows", "remove_all_windows", "drop_all_users"},
    "sampling": {f"{mode}_bs{batch}" for mode in ("random", "dates", "individuals", "all") for batch in (16, 64)},
    "normalizations": {"identity", "standard", "min-max", "in-min-max", "instance", "revin"},
    "reference": {"persistence", "patchtst_proxy", "chronos2_proxy"},
    "losses": {"mse", "mae", "nmse", "nmae", "relative_mse"},
    "linear_models": {"persistence", "periodic", "linear_adam", "numpy_lstsq", "ridge"},
    "central_per_user": {"central", "per_user"},
}


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    user: np.ndarray
    query_t: np.ndarray

    def take(self, indices: np.ndarray) -> "Windows":
        return Windows(
            self.x[indices],
            self.y[indices],
            self.user[indices],
            self.query_t[indices],
        )


def roots() -> tuple[Path, Path]:
    project = Path(__file__).resolve().parents[2]
    workspace = project.parents[2]
    return project, workspace


def load_archive_generator(workspace: Path) -> dict[str, Any]:
    """Execute the generator definitions and simple-population config in the archive notebook."""
    notebook = json.loads((workspace / "archive" / "synthetic_generator.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
        "copy": copy,
        "dataclass": dataclass,
        "Dict": Dict,
        "Any": Any,
        "List": List,
        "Union": Union,
    }
    for cell_index in (4, 6, 8):
        exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    config_source = "".join(notebook["cells"][13]["source"]).split("simple_population =", 1)[0]
    exec(config_source, namespace)
    return namespace


def generate_dataset(project: Path, workspace: Path) -> tuple[np.ndarray, pd.DataFrame]:
    namespace = load_archive_generator(workspace)
    cluster_a = copy.deepcopy(namespace["simple_cluster_a_config"])
    cluster_b = copy.deepcopy(namespace["simple_cluster_b_config"])
    cluster_a["n_series"] = 8
    cluster_b["n_series"] = 8
    population = namespace["Population"].from_config({"clusters": [cluster_a, cluster_b]})
    values, metadata = population.generate(np.arange(360), seed=42)

    # Controlled plateaus make the constant-window policies observable.
    values[0, 60:96] = values[0, 60]
    values[1, 300:336] = values[1, 300]

    dataset_dir = project / "datasets" / "synthetic_smoke"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(values.T, columns=[f"series_{i:02d}" for i in range(len(values))])
    frame.insert(0, "date_index", np.arange(values.shape[1]))
    frame.to_csv(dataset_dir / "synthetic_smoke.csv", index=False)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    (dataset_dir / "config.json").write_text(
        json.dumps(
            {
                "generator": "archive/synthetic_generator.ipynb",
                "generator_seed": 42,
                "n_series_per_cluster": 8,
                "n_steps": 360,
                "date_col": "date_index",
                "injected_plateaus": [[0, 60, 96], [1, 300, 336]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return values, metadata


def make_windows(values: np.ndarray, target_start: int, target_end: int) -> Windows:
    """Build ``X=(t-L,t]``/``Y=(t,t+H]`` with targets inside the period."""
    xs, ys, users, query_dates = [], [], [], []
    first = max(LAGS - 1, int(target_start) - 1)
    last = min(values.shape[1] - HORIZON - 1, int(target_end) - HORIZON - 1)
    for user, series in enumerate(values):
        for query_t in range(first, last + 1):
            xs.append(series[query_t - LAGS + 1 : query_t + 1])
            ys.append(series[query_t + 1 : query_t + HORIZON + 1])
            users.append(user)
            query_dates.append(query_t)
    return Windows(
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(users),
        np.asarray(query_dates),
    )


def constant_mask(windows: Windows) -> np.ndarray:
    return (~np.isfinite(windows.x).all(axis=1)) | (np.std(windows.x, axis=1) <= 1e-12)


def filter_windows(windows: Windows, remove_windows: bool, drop_users: bool) -> Windows:
    bad = constant_mask(windows)
    keep = np.ones(len(windows.x), dtype=bool)
    if remove_windows:
        keep &= ~bad
    if drop_users:
        keep &= ~np.isin(windows.user, np.unique(windows.user[bad]))
    return windows.take(np.flatnonzero(keep))


def sample_training(windows: Windows, mode: str, batch_size: int, rng: np.random.Generator) -> Windows:
    if mode == "all":
        return windows
    if mode == "random":
        count = min(max(batch_size * 8, 128), len(windows.x))
        return windows.take(rng.choice(len(windows.x), size=count, replace=False))
    if mode == "dates":
        dates = np.unique(windows.query_t)[::4]
        return windows.take(np.flatnonzero(np.isin(windows.query_t, dates)))
    if mode == "individuals":
        users = np.unique(windows.user)
        chosen = rng.choice(users, size=max(1, len(users) // 2), replace=False)
        return windows.take(np.flatnonzero(np.isin(windows.user, chosen)))
    raise ValueError(mode)


def normalization_state(x: np.ndarray, name: str) -> dict[str, float]:
    if name == "standard":
        return {"offset": float(x.mean()), "scale": float(max(x.std(), 1e-8))}
    if name == "min-max":
        low, high = float(x.min()), float(x.max())
        return {"offset": low, "scale": max(high - low, 1e-8)}
    return {}


def normalize(x: np.ndarray, name: str, state: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name in {"identity", "none"}:
        offset = np.zeros((len(x), 1))
        scale = np.ones((len(x), 1))
    elif name in {"standard", "min-max"}:
        offset = np.full((len(x), 1), state["offset"])
        scale = np.full((len(x), 1), state["scale"])
    elif name in {"instance", "revin"}:
        offset = x.mean(axis=1, keepdims=True)
        raw_scale = x.std(axis=1, keepdims=True)
        scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
    elif name == "in-min-max":
        offset = x.min(axis=1, keepdims=True)
        raw_scale = x.max(axis=1, keepdims=True) - offset
        scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
    else:
        raise ValueError(name)
    return (x - offset) / scale, offset, scale


def objective_and_gradient(prediction: np.ndarray, target: np.ndarray, x: np.ndarray, loss: str) -> tuple[float, np.ndarray]:
    error = prediction - target
    count = error.size
    if loss == "mse":
        return float(np.mean(error**2)), 2 * error / count
    if loss == "mae":
        return float(np.mean(np.abs(error))), np.sign(error) / count
    if loss in {"nmse", "nmae", "relative_mse"}:
        if loss == "relative_mse":
            scale = np.maximum(np.abs(x.mean(axis=1, keepdims=True)), 1e-4)
        else:
            scale = np.maximum(x.std(axis=1, keepdims=True), 1e-4)
        if loss == "nmae":
            return float(np.mean(np.abs(error) / scale)), np.sign(error) / scale / count
        return float(np.mean((error / scale) ** 2)), 2 * error / (scale**2) / count
    raise ValueError(loss)


def predict(weights: np.ndarray, x: np.ndarray, normalization: str, state: dict[str, float]) -> np.ndarray:
    x_norm, offset, scale = normalize(x, normalization, state)
    design = np.column_stack([x_norm, np.ones(len(x_norm))])
    return (design @ weights) * scale + offset


def fit_adam(
    train: Windows,
    valid: Windows,
    *,
    seed: int,
    normalization: str,
    loss: str,
    batch_size: int,
    epochs: int = EPOCHS,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    state = normalization_state(train.x, normalization)
    x_norm, offset, scale = normalize(train.x, normalization, state)
    design = np.column_stack([x_norm, np.ones(len(x_norm))])
    weights = np.zeros((design.shape[1], HORIZON))
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    step = 0
    history = []

    for epoch in range(1, epochs + 1):
        objectives = []
        for begin in range(0, len(train.x), batch_size):
            indices = rng.permutation(len(train.x))[begin : begin + batch_size]
            if len(indices) == 0:
                continue
            prediction_norm = design[indices] @ weights
            prediction = prediction_norm * scale[indices] + offset[indices]
            objective, gradient = objective_and_gradient(prediction, train.y[indices], train.x[indices], loss)
            gradient_norm = gradient * scale[indices]
            grad_weights = design[indices].T @ gradient_norm
            grad_norm = np.linalg.norm(grad_weights)
            if grad_norm > 50:
                grad_weights *= 50 / grad_norm
            step += 1
            first = 0.9 * first + 0.1 * grad_weights
            second = 0.999 * second + 0.001 * grad_weights**2
            first_hat = first / (1 - 0.9**step)
            second_hat = second / (1 - 0.999**step)
            weights -= 0.01 * first_hat / (np.sqrt(second_hat) + 1e-8)
            objectives.append(objective)
        valid_prediction = predict(weights, valid.x, normalization, state)
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(np.mean(objectives)),
                "valid_mse": float(np.mean((valid_prediction - valid.y) ** 2)),
            }
        )
    return weights, state, history


def fit_closed_form(train: Windows, normalization: str, ridge: float = 0.0) -> tuple[np.ndarray, dict[str, float]]:
    state = normalization_state(train.x, normalization)
    x_norm, offset, scale = normalize(train.x, normalization, state)
    design = np.column_stack([x_norm, np.ones(len(x_norm))])
    target = (train.y - offset) / scale
    if ridge == 0:
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        penalty = np.eye(design.shape[1]) * ridge
        penalty[-1, -1] = 0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return weights, state


def metrics(prediction: np.ndarray, windows: Windows) -> dict[str, float]:
    squared_errors = (prediction - windows.y) ** 2
    row_mse = np.mean(squared_errors, axis=1)
    user_mse = np.asarray([row_mse[windows.user == user].mean() for user in np.unique(windows.user)])
    worst_count = max(1, math.ceil(0.1 * len(user_mse)))
    return {
        "mse": float(squared_errors.mean()),
        "std_mse": float(squared_errors.std()),
        "user_mse": float(user_mse.mean()),
        "std_user_mse": float(user_mse.std()),
        "w10_mse": float(np.sort(user_mse)[-worst_count:].mean()),
    }


def trained_run(
    train: Windows,
    valid: Windows,
    test: Windows,
    *,
    seed: int,
    mode: str = "random",
    batch_size: int = 32,
    normalization: str = "instance",
    loss: str = "mse",
) -> tuple[dict[str, float], list[dict[str, float]], int]:
    sampled = sample_training(train, mode, batch_size, np.random.default_rng(seed))
    weights, state, history = fit_adam(
        sampled,
        valid,
        seed=seed,
        normalization=normalization,
        loss=loss,
        batch_size=batch_size,
    )
    return metrics(predict(weights, test.x, normalization, state), test), history, len(sampled.x)


def add_result(rows: list[dict[str, Any]], family: str, method: str, seed: int, result: dict[str, float], n_train: int, n_test: int) -> None:
    rows.append({"family": family, "method": method, "seed": seed, **result, "n_train": n_train, "n_test": n_test})


def run_benchmark(values: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = make_windows(values, 0, 216)
    valid = make_windows(values, 216, 288)
    test = make_windows(values, 288, 360)
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []

    constant_policies = {
        "keep": (False, False, False, False),
        "remove_train_windows": (True, False, False, False),
        "remove_eval_windows": (False, False, True, False),
        "remove_all_windows": (True, False, True, False),
        "drop_all_users": (False, True, False, True),
    }
    for method, (remove_train, drop_train, remove_eval, drop_eval) in constant_policies.items():
        train_policy = filter_windows(train, remove_train, drop_train)
        valid_policy = filter_windows(valid, remove_eval, drop_eval)
        test_policy = filter_windows(test, remove_eval, drop_eval)
        for seed in SEEDS:
            result, history, n_train = trained_run(train_policy, valid_policy, test_policy, seed=seed)
            add_result(rows, "constants", method, seed, result, n_train, len(test_policy.x))

    for mode in ("random", "dates", "individuals", "all"):
        for batch_size in (16, 64):
            method = f"{mode}_bs{batch_size}"
            for seed in SEEDS:
                result, history, n_train = trained_run(train, valid, test, seed=seed, mode=mode, batch_size=batch_size)
                add_result(rows, "sampling", method, seed, result, n_train, len(test.x))

    for loss in ("mse", "mae", "nmse", "nmae", "relative_mse"):
        for seed in SEEDS:
            result, history, n_train = trained_run(train, valid, test, seed=seed, loss=loss)
            add_result(rows, "losses", loss, seed, result, n_train, len(test.x))
            if loss == "mse" and seed == SEEDS[0]:
                histories.extend({"family": "losses", "method": loss, "seed": seed, **item} for item in history)

    for normalization in ("identity", "standard", "min-max", "in-min-max", "instance", "revin"):
        for seed in SEEDS:
            result, history, n_train = trained_run(train, valid, test, seed=seed, normalization=normalization, loss="nmse")
            add_result(rows, "normalizations", normalization, seed, result, n_train, len(test.x))

    # The publication reference family compares persistence, PatchTST, and
    # Chronos-2. Keep this smoke dependency-light by exercising the same
    # three-column reporting contract with explicit NumPy proxies for the two
    # unavailable backbones.
    for seed in SEEDS:
        persistence = np.repeat(test.x[:, -1:], HORIZON, axis=1)
        add_result(
            rows,
            "reference",
            "persistence",
            seed,
            metrics(persistence, test),
            len(train.x),
            len(test.x),
        )
        patchtst_result, _, n_train = trained_run(
            train,
            valid,
            test,
            seed=seed,
            normalization="instance",
            loss="nmse",
        )
        add_result(
            rows,
            "reference",
            "patchtst_proxy",
            seed,
            patchtst_result,
            n_train,
            len(test.x),
        )
        chronos2_proxy = test.x[:, :HORIZON]
        add_result(
            rows,
            "reference",
            "chronos2_proxy",
            seed,
            metrics(chronos2_proxy, test),
            len(train.x),
            len(test.x),
        )

    for seed in SEEDS:
        persistence = np.repeat(test.x[:, -1:], HORIZON, axis=1)
        add_result(rows, "linear_models", "persistence", seed, metrics(persistence, test), len(train.x), len(test.x))
        periodic = test.x[:, :HORIZON]
        add_result(rows, "linear_models", "periodic", seed, metrics(periodic, test), len(train.x), len(test.x))

        result, _, n_train = trained_run(train, valid, test, seed=seed, mode="all")
        add_result(rows, "linear_models", "linear_adam", seed, result, n_train, len(test.x))
        for method, ridge in (("numpy_lstsq", 0.0), ("ridge", 1.0)):
            weights, state = fit_closed_form(train, "instance", ridge=ridge)
            add_result(rows, "linear_models", method, seed, metrics(predict(weights, test.x, "instance", state), test), len(train.x), len(test.x))

    for seed in SEEDS:
        result, _, n_train = trained_run(train, valid, test, seed=seed, mode="all")
        add_result(rows, "central_per_user", "central", seed, result, n_train, len(test.x))

        prediction = np.empty_like(test.y)
        total_train = 0
        for user in np.unique(test.user):
            train_user = train.take(np.flatnonzero(train.user == user))
            valid_user = valid.take(np.flatnonzero(valid.user == user))
            test_indices = np.flatnonzero(test.user == user)
            test_user = test.take(test_indices)
            weights, state, _ = fit_adam(
                train_user,
                valid_user,
                seed=seed,
                normalization="instance",
                loss="mse",
                batch_size=32,
            )
            prediction[test_indices] = predict(weights, test_user.x, "instance", state)
            total_train += len(train_user.x)
        add_result(rows, "central_per_user", "per_user", seed, metrics(prediction, test), total_train, len(test.x))

    results = pd.DataFrame(rows)
    observed_families = set(results["family"])
    expected_families = set(BENCHMARK_FAMILIES)
    if observed_families != expected_families:
        missing = sorted(expected_families - observed_families)
        unexpected = sorted(observed_families - expected_families)
        raise RuntimeError(
            f"synthetic benchmark family mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return results, pd.DataFrame(histories)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["family", "method"], sort=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            w10_mean=("w10_mse", "mean"),
            w10_std=("w10_mse", "std"),
            n_train=("n_train", "mean"),
            n_test=("n_test", "mean"),
        )
        .reset_index()
    )


def write_manifest_runs(results: pd.DataFrame, project: Path) -> list[SelectedRun]:
    """Persist the synthetic smoke matrix through the current run contract."""
    selected: list[SelectedRun] = []
    dataset_config = project / "datasets" / "synthetic_smoke" / "config.json"
    for (family, method), frame in results.groupby(["family", "method"], sort=False):
        root = identity_path(
            project / "outputs" / "synthetic_smoke" / family,
            "synthetic_smoke",
            LAGS,
            HORIZON,
            "numpy_linear_proxy",
            ["method"],
            {"method": method},
        )
        allocation = allocate_run(
            root,
            project="timetensors",
            workflow=f"synthetic_smoke/{family}",
            dataset="synthetic_smoke",
            lookback=LAGS,
            horizon=HORIZON,
            backbone="numpy_linear_proxy",
            model_config_order=["method"],
            model_config={"method": method},
            pipeline_config={
                "epochs": EPOCHS,
                "dataset_generator_seed": 42,
                "n_users": 16,
                "n_steps": 360,
            },
            runtime_config={"implementation": "numpy"},
            seeds=list(SEEDS),
            purpose="smoke",
            mode="test",
            display_name=method,
            row_config=["method"],
            inputs={"dataset_config": str(dataset_config)},
            policy="overwrite_exact",
            skip_completed=True,
            launch_id="synthetic-smoke",
        )
        if allocation.action != "skip":
            mark_status(allocation.run_dir, "running")
            required = []
            for row in frame.to_dict(orient="records"):
                seed = int(row["seed"])
                relative = f"seed_{seed}/metrics.json"
                path = allocation.run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                mark_status(allocation.run_dir, "completed", required_artifacts=[relative], seed=seed)
                required.append(relative)
            mark_status(allocation.run_dir, "completed", required_artifacts=required)
        manifest = load_manifest(allocation.run_dir)
        selected.append(SelectedRun(allocation.run_dir, manifest, method))
    return selected


def scaled(value: float, deviation: float, exponent: int) -> str:
    factor = 10.0**exponent
    return f"${value / factor:.2f} \\pm {deviation / factor:.2f}$"


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def write_tables(summary: pd.DataFrame, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for family, frame in summary.groupby("family", sort=False):
        lines = [
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Method & Scale & MSE & W10 MSE & Train windows \\\\",
            "\\midrule",
        ]
        for row in frame.itertuples(index=False):
            exponent = 0 if row.mse_mean == 0 else math.floor(math.log10(abs(row.mse_mean)))
            multiplier = f"$\\times 10^{{{exponent}}}$"
            mse = scaled(row.mse_mean, row.mse_std, exponent)
            w10 = scaled(row.w10_mean, row.w10_std, exponent)
            lines.append(f"{latex_escape(row.method)} & {multiplier} & {mse} & {w10} & {row.n_train:.0f} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", ""])
        (tables / f"table_{family}.tex").write_text("\n".join(lines), encoding="utf-8")

    markdown = [
        "# Synthetic smoke benchmark",
        "",
        "Archived two-cluster generator; 16 users, 360 steps, L=24, H=6, six epochs, two seeds.",
        "",
        "| Family | Method | MSE (mean ± std) | W10 MSE (mean ± std) | Train | Test |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        markdown.append(
            f"| {row.family} | {row.method} | {row.mse_mean:.4g} ± {row.mse_std:.3g} | "
            f"{row.w10_mean:.4g} ± {row.w10_std:.3g} | {row.n_train:.0f} | {row.n_test:.0f} |"
        )
    (output / "results_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def polyline(values: np.ndarray, left: float, top: float, width: float, height: float, log: bool = False) -> str:
    values = np.asarray(values, dtype=float)
    if log:
        values = np.log10(np.maximum(values, 1e-12))
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = left + width * index / max(len(values) - 1, 1)
        y = top + height * (high - value) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def write_series_plot(values: np.ndarray, output: Path) -> None:
    width, height = 900, 420
    colors = ("#2563eb", "#0f766e", "#d97706", "#dc2626")
    selected = (0, 2, 8, 10)
    low, high = float(values[list(selected)].min()), float(values[list(selected)].max())
    span = high - low
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="28" font-family="sans-serif" font-size="18" font-weight="600">Archived synthetic population</text>',
        '<line x1="55" y1="365" x2="870" y2="365" stroke="#64748b"/>',
        '<line x1="55" y1="45" x2="55" y2="365" stroke="#64748b"/>',
    ]
    for index, user in enumerate(selected):
        points = []
        for step, value in enumerate(values[user]):
            x = 55 + 815 * step / (values.shape[1] - 1)
            y = 45 + 320 * (high - value) / span
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[index]}" stroke-width="1.4" opacity="0.85"/>')
        lines.append(f'<text x="{650 + 105 * (index % 2)}" y="{25 + 18 * (index // 2)}" fill="{colors[index]}" font-family="sans-serif" font-size="12">series {user}</text>')
    for split, label in ((216, "train/valid"), (288, "valid/test")):
        x = 55 + 815 * split / (values.shape[1] - 1)
        lines.append(f'<line x1="{x:.1f}" y1="45" x2="{x:.1f}" y2="365" stroke="#94a3b8" stroke-dasharray="5 4"/>')
        lines.append(f'<text x="{x + 5:.1f}" y="62" font-family="sans-serif" font-size="11" fill="#475569">{label}</text>')
    lines.extend([
        '<text x="450" y="402" text-anchor="middle" font-family="sans-serif" font-size="12">time step</text>',
        '<text x="16" y="205" transform="rotate(-90 16 205)" text-anchor="middle" font-family="sans-serif" font-size="12">value</text>',
        '</svg>',
    ])
    (output / "synthetic_series.svg").write_text("\n".join(lines), encoding="utf-8")


def write_training_plot(history: pd.DataFrame, output: Path) -> None:
    train_values = np.log10(np.maximum(history["train_objective"].to_numpy(), 1e-12))
    valid_values = np.log10(np.maximum(history["valid_mse"].to_numpy(), 1e-12))
    low = float(min(train_values.min(), valid_values.min()))
    high = float(max(train_values.max(), valid_values.max()))
    span = max(high - low, 1e-9)

    def shared_points(values: np.ndarray) -> str:
        points = []
        for index, value in enumerate(values):
            x = 70 + 760 * index / max(len(values) - 1, 1)
            y = 55 + 280 * (high - value) / span
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    train_points = shared_points(train_values)
    valid_points = shared_points(valid_values)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="400" viewBox="0 0 900 400">
<rect width="100%" height="100%" fill="white"/>
<text x="70" y="28" font-family="sans-serif" font-size="18" font-weight="600">Linear-model training, MSE, seed 1</text>
<line x1="70" y1="335" x2="830" y2="335" stroke="#64748b"/><line x1="70" y1="55" x2="70" y2="335" stroke="#64748b"/>
<polyline points="{train_points}" fill="none" stroke="#2563eb" stroke-width="3"/><polyline points="{valid_points}" fill="none" stroke="#d97706" stroke-width="3"/>
<text x="680" y="30" fill="#2563eb" font-family="sans-serif" font-size="12">average train objective</text><text x="680" y="47" fill="#d97706" font-family="sans-serif" font-size="12">validation MSE</text>
<text x="450" y="380" text-anchor="middle" font-family="sans-serif" font-size="12">epoch (1–6)</text><text x="18" y="195" transform="rotate(-90 18 195)" text-anchor="middle" font-family="sans-serif" font-size="12">log10 metric</text>
</svg>'''
    (output / "training_curves.svg").write_text(svg, encoding="utf-8")


def write_family_plot(summary: pd.DataFrame, output: Path) -> None:
    rows = len(summary) + summary["family"].nunique()
    height = 90 + rows * 25
    values = np.log10(np.maximum(summary["mse_mean"].to_numpy(), 1e-12))
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-9)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="35" y="30" font-family="sans-serif" font-size="19" font-weight="600">Synthetic benchmark test MSE</text>',
        '<text x="35" y="50" font-family="sans-serif" font-size="12" fill="#475569">Bar length uses log10(MSE); labels report the two-seed mean ± standard deviation.</text>',
    ]
    y = 78
    colors = ["#2563eb", "#0f766e", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]
    for family_index, (family, frame) in enumerate(summary.groupby("family", sort=False)):
        svg.append(f'<text x="35" y="{y}" font-family="sans-serif" font-size="14" font-weight="600">{html.escape(family.replace("_", " "))}</text>')
        y += 19
        for row in frame.itertuples(index=False):
            log_value = math.log10(max(row.mse_mean, 1e-12))
            bar_width = 80 + 420 * (log_value - low) / span
            svg.append(f'<text x="55" y="{y + 12}" font-family="sans-serif" font-size="11">{html.escape(row.method)}</text>')
            svg.append(f'<rect x="260" y="{y}" width="{bar_width:.1f}" height="15" rx="3" fill="{colors[family_index % len(colors)]}" opacity="0.82"/>')
            svg.append(f'<text x="{270 + bar_width:.1f}" y="{y + 12}" font-family="monospace" font-size="11">{row.mse_mean:.3g} ± {row.mse_std:.2g}</text>')
            y += 22
        y += 9
    svg.append("</svg>")
    (output / "family_mse.svg").write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    project, workspace = roots()
    output = project / "outputs" / "reports" / "synthetic_smoke"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    values, metadata = generate_dataset(project, workspace)
    results, history = run_benchmark(values)
    summary = aggregate(results)
    selected = write_manifest_runs(results, project)

    results.to_csv(output / "results_by_seed.csv", index=False)
    summary.to_csv(output / "results_summary.csv", index=False)
    history.to_csv(output / "training_history.csv", index=False)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "lags": LAGS,
                "horizon": HORIZON,
                "epochs": EPOCHS,
                "n_users": int(values.shape[0]),
                "n_steps": int(values.shape[1]),
                "clusters": metadata["cluster"].value_counts().to_dict(),
                "implementation": "NumPy Adam linear smoke runner",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_tables(summary, output)
    write_series_plot(values, plots)
    write_training_plot(history, plots)
    write_family_plot(summary, plots)
    write_report_manifest(
        output / "report_manifest.json",
        inputs=selected,
        config_policy="distinct",
        repeat_policy="selected",
        filters={"mode": "test", "purposes": ["smoke"], "families": list(BENCHMARK_FAMILIES)},
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
