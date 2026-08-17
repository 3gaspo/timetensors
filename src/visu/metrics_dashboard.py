"""Interactive metrics dashboard helpers for TimeTensor outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def configure_pandas() -> None:
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 80)


def tensor_mean(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().mean().cpu())
    if isinstance(value, dict):
        values = [tensor_mean(item) for item in value.values()]
        values = [item for item in values if not math.isnan(item)]
        return float(np.mean(values)) if values else float("nan")
    if isinstance(value, (list, tuple)):
        values = [tensor_mean(item) for item in value]
        values = [item for item in values if not math.isnan(item)]
        return float(np.mean(values)) if values else float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def discover_losses(output_root: str | Path) -> pd.DataFrame:
    root = Path(output_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(columns=["dataset", "setting", "model", "split", "metric", "value", "path"])

    for loss_path in root.rglob("all_losses.pt"):
        try:
            relative_path = loss_path.relative_to(root)
        except ValueError:
            continue
        parts = relative_path.parts
        if len(parts) < 4:
            continue
        dataset, setting, model = parts[-4], parts[-3], parts[-2]
        try:
            data = torch.load(loss_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(loss_path, map_location="cpu")
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "model": model,
                    "split": "<load_error>",
                    "metric": type(exc).__name__,
                    "value": float("nan"),
                    "path": str(loss_path),
                }
            )
            continue
        for split, payload in (data or {}).items():
            if not isinstance(payload, dict):
                continue
            metrics = {
                **dict(payload.get("losses") or {}),
                **dict(payload.get("summaries") or {}),
            }
            for metric, value in metrics.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "model": model,
                        "split": split,
                        "metric": metric,
                        "value": tensor_mean(value),
                        "path": str(loss_path),
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "setting", "model", "split", "metric", "value", "path"])


def latex_escape(text: Any) -> str:
    return str(text).replace("_", r"\_")


def is_best(a: float, b: float, tol: float = 1e-12) -> bool:
    return not np.isnan(a) and not np.isnan(b) and abs(a - b) <= tol


def improvement_pct(ref: float, cur: float, lower_is_better: bool = True) -> float:
    if np.isnan(ref) or np.isnan(cur) or ref == 0:
        return float("nan")
    if lower_is_better:
        return (ref - cur) / ref * 100.0
    return (cur - ref) / ref * 100.0


def selected_metric_frame(
    data: pd.DataFrame,
    datasets: tuple[str, ...],
    settings: tuple[str, ...],
    models: tuple[str, ...],
    split: str,
    metric: str,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    frame = data[
        data["dataset"].isin(datasets)
        & data["setting"].isin(settings)
        & data["model"].isin(models)
        & (data["split"] == split)
        & (data["metric"] == metric)
    ].copy()
    frame["value"] = frame["value"] * float(multiplier)
    return frame


def make_metric_table(
    frame: pd.DataFrame,
    datasets: tuple[str, ...],
    settings: tuple[str, ...],
    models: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["Dataset", "Setting"]), columns=list(models))
    pivot = frame.pivot_table(index=["dataset", "setting"], columns="model", values="value", aggfunc="mean")
    index = pd.MultiIndex.from_product([datasets, settings], names=["Dataset", "Setting"])
    return pivot.reindex(index=index, columns=list(models))


def table_with_improvements(table: pd.DataFrame, ref_model: str, lower_is_better: bool = True) -> pd.DataFrame:
    rows = []
    index = []
    if table.empty:
        return table
    for dataset, dataset_table in table.groupby(level=0, sort=False):
        for key, row in dataset_table.iterrows():
            index.append(key)
            rows.append(row)
        if ref_model in dataset_table.columns:
            improvements = []
            for model in dataset_table.columns:
                values = [
                    improvement_pct(row[ref_model], row[model], lower_is_better)
                    for _, row in dataset_table.iterrows()
                ]
                values = [value for value in values if not np.isnan(value)]
                improvements.append(float(np.mean(values)) if values else float("nan"))
            index.append((dataset, "Improvement (%)"))
            rows.append(pd.Series(improvements, index=dataset_table.columns))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=table.index.names), columns=table.columns)


def render_html_table(table: pd.DataFrame, decimals: int, lower_is_better: bool, ref_model: str):
    from IPython.display import HTML

    del ref_model

    def fmt(value):
        if pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}"

    def highlight(row):
        styles = ["" for _ in row]
        is_imp = row.name[1] == "Improvement (%)"
        valid = row.dropna()
        if valid.empty:
            return styles
        best = valid.max() if is_imp or not lower_is_better else valid.min()
        return ["font-weight: bold" if pd.notna(value) and is_best(float(value), float(best)) else "" for value in row]

    styled = table.style.format(fmt).apply(highlight, axis=1)
    return HTML(styled.to_html())


def generate_results_latex(
    table: pd.DataFrame,
    metric_key: str,
    output_tex_path: str | Path | None = None,
    lower_is_better: bool = True,
    decimals: int = 4,
    ref_model: str | None = None,
) -> str:
    if table.empty:
        latex = "% Empty metrics table"
        if output_tex_path:
            Path(output_tex_path).write_text(latex, encoding="utf-8")
        return latex

    model_names = list(table.columns)
    datasets = list(dict.fromkeys(table.index.get_level_values(0)))
    settings = list(dict.fromkeys(table.index.get_level_values(1)))
    ref_model = ref_model or model_names[0]

    def fmt_value(value: float) -> str:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}"

    def fmt_imp(value: float) -> str:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}\\%"

    def avg_improvements(rows: pd.DataFrame) -> list[float]:
        out = []
        for model in model_names:
            if model == ref_model:
                out.append(0.0)
                continue
            values = [
                improvement_pct(row[ref_model], row[model], lower_is_better)
                for _, row in rows.iterrows()
            ]
            values = [value for value in values if not np.isnan(value)]
            out.append(float(np.mean(values)) if values else float("nan"))
        return out

    lines = [
        r"\begin{table}[h!]",
        fr"\caption{{Results for {latex_escape(metric_key)} metric.}}",
        r"\vspace{-4mm}",
        r"\centering",
        r"\scalebox{0.4}{",
        fr"\begin{{tabular}}{{{'lc' + 'c' * len(model_names)}}}",
        r"\toprule",
    ]
    header = ["", "Setting"] + [fr"\thead{{{latex_escape(model)}}}" for model in model_names]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for dataset_index, dataset in enumerate(datasets):
        dataset_table = table.loc[dataset]
        row_count = len(dataset_table)
        for row_index, (setting, row) in enumerate(dataset_table.iterrows()):
            dataset_cell = fr"\multirow{{{row_count}}}{{*}}{{{latex_escape(dataset)}}}" if row_index == 0 else ""
            values = row.to_numpy(dtype=float)
            valid = values[~np.isnan(values)]
            best = None if len(valid) == 0 else (np.min(valid) if lower_is_better else np.max(valid))
            cells = []
            for value in values:
                cell = fmt_value(value)
                if best is not None and is_best(value, best):
                    cell = fr"\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(" & ".join([dataset_cell, str(setting).replace("_", "-")] + cells) + r" \\")

        improvements = avg_improvements(dataset_table)
        best_imp = np.nanmax(improvements) if not np.all(np.isnan(improvements)) else float("nan")
        imp_cells = []
        for improvement in improvements:
            cell = fmt_imp(improvement)
            if is_best(improvement, best_imp):
                cell = fr"\textbf{{{cell}}}"
            imp_cells.append(cell)
        lines.append(" & ".join(["", r"\textit{Improvement}"] + imp_cells) + r" \\")
        if dataset_index < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\midrule")
    lines.append(r"\midrule")
    for setting_index, setting in enumerate(settings):
        rows = table.xs(setting, level=1, drop_level=False)
        improvements = avg_improvements(rows)
        best_imp = np.nanmax(improvements) if not np.all(np.isnan(improvements)) else float("nan")
        imp_cells = []
        for improvement in improvements:
            cell = fmt_imp(improvement)
            if is_best(improvement, best_imp):
                cell = fr"\textbf{{{cell}}}"
            imp_cells.append(cell)
        lines.append(
            " & ".join(["Improvements" if setting_index == 0 else "", str(setting).replace("_", "-")] + imp_cells)
            + r" \\"
        )

    lines.append(r"\midrule")
    overall = avg_improvements(table)
    best_overall = np.nanmax(overall) if not np.all(np.isnan(overall)) else float("nan")
    overall_cells = []
    for improvement in overall:
        cell = fmt_imp(improvement)
        if is_best(improvement, best_overall):
            cell = fr"\textbf{{{cell}}}"
        overall_cells.append(cell)
    lines.append(" & ".join(["Overall improvements", ""] + overall_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\label{tab:main}", r"\end{table}"])

    latex = "\n".join(lines)
    if output_tex_path:
        out = Path(output_tex_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex, encoding="utf-8")
    return latex


def display_dashboard(default_output_root: str | Path | None = None) -> None:
    import ipywidgets as widgets
    from IPython.display import Latex, clear_output, display

    configure_pandas()
    output_root = widgets.Text(
        value=str(default_output_root or (Path.cwd() / "outputs")),
        description="Output root",
        layout=widgets.Layout(width="90%"),
    )
    refresh_button = widgets.Button(description="Refresh", button_style="info")
    dataset_select = widgets.SelectMultiple(description="Datasets", options=[], layout=widgets.Layout(width="32%", height="160px"))
    setting_select = widgets.SelectMultiple(description="L-H", options=[], layout=widgets.Layout(width="32%", height="160px"))
    model_select = widgets.SelectMultiple(description="Models", options=[], layout=widgets.Layout(width="32%", height="160px"))
    split_dropdown = widgets.Dropdown(description="Split", options=[])
    metric_dropdown = widgets.Dropdown(description="Metric", options=[])
    ref_dropdown = widgets.Dropdown(description="Ref", options=[])
    lower_is_better = widgets.Checkbox(value=True, description="Lower is better")
    decimals = widgets.IntSlider(value=4, min=0, max=8, description="Decimals")
    multiplier = widgets.FloatText(value=1.0, description="Multiplier")
    tex_path = widgets.Text(value="results_table.tex", description=".tex path", layout=widgets.Layout(width="70%"))
    export_button = widgets.Button(description="Export .tex", button_style="success")
    status = widgets.Output()
    table_output = widgets.Output()
    latex_output = widgets.Output()
    state = {"data": pd.DataFrame(), "table": pd.DataFrame(), "latex": ""}

    def refresh_options(_=None):
        data = discover_losses(output_root.value)
        state["data"] = data
        datasets = sorted(data["dataset"].dropna().unique().tolist()) if not data.empty else []
        settings = sorted(data["setting"].dropna().unique().tolist()) if not data.empty else []
        models = sorted(data["model"].dropna().unique().tolist()) if not data.empty else []
        splits = sorted(data["split"].dropna().unique().tolist()) if not data.empty else []
        metrics = sorted(data["metric"].dropna().unique().tolist()) if not data.empty else []

        dataset_select.options = datasets
        dataset_select.value = tuple(datasets)
        setting_select.options = settings
        setting_select.value = tuple(settings)
        model_select.options = models
        model_select.value = tuple(models)
        split_dropdown.options = splits
        split_dropdown.value = splits[0] if splits else None
        metric_dropdown.options = metrics
        metric_dropdown.value = metrics[0] if metrics else None
        ref_dropdown.options = models
        ref_dropdown.value = models[0] if models else None

        with status:
            clear_output()
            print(f"Discovered {len(data)} metric rows from {output_root.value}")
        render_table()

    def render_table(_=None):
        data = state["data"]
        selected = selected_metric_frame(
            data,
            tuple(dataset_select.value),
            tuple(setting_select.value),
            tuple(model_select.value),
            split_dropdown.value,
            metric_dropdown.value,
            multiplier.value,
        )
        table = make_metric_table(
            selected,
            tuple(dataset_select.value),
            tuple(setting_select.value),
            tuple(model_select.value),
        )
        state["table"] = table
        state["latex"] = generate_results_latex(
            table,
            metric_key=metric_dropdown.value or "metric",
            lower_is_better=lower_is_better.value,
            decimals=decimals.value,
            ref_model=ref_dropdown.value,
        )
        display_table = table_with_improvements(table, ref_dropdown.value, lower_is_better.value)
        with table_output:
            clear_output()
            if table.empty:
                print("No matching metrics found.")
            else:
                display(render_html_table(display_table, decimals.value, lower_is_better.value, ref_dropdown.value))
        with latex_output:
            clear_output()
            display(Latex(state["latex"]))

    def export_tex(_=None):
        if state["table"].empty:
            with status:
                clear_output()
                print("Nothing to export: current table is empty.")
            return
        out = Path(tex_path.value).expanduser()
        if not out.is_absolute():
            out = Path(output_root.value).expanduser().resolve() / out
        generate_results_latex(
            state["table"],
            metric_key=metric_dropdown.value or "metric",
            output_tex_path=out,
            lower_is_better=lower_is_better.value,
            decimals=decimals.value,
            ref_model=ref_dropdown.value,
        )
        with status:
            clear_output()
            print(f"Exported LaTeX table to {out}")

    refresh_button.on_click(refresh_options)
    export_button.on_click(export_tex)
    for widget in [
        dataset_select,
        setting_select,
        model_select,
        split_dropdown,
        metric_dropdown,
        ref_dropdown,
        lower_is_better,
        decimals,
        multiplier,
    ]:
        widget.observe(render_table, names="value")

    controls = widgets.VBox(
        [
            widgets.HBox([output_root, refresh_button]),
            widgets.HBox([dataset_select, setting_select, model_select]),
            widgets.HBox([split_dropdown, metric_dropdown, ref_dropdown, lower_is_better, decimals, multiplier]),
            widgets.HBox([tex_path, export_button]),
            status,
        ]
    )
    display(controls, table_output, latex_output)
    refresh_options()
