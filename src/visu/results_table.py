"""Build publication-ready LaTeX tables from TimeTensor experiment losses.

Expected experiment layout::

    EXPERIMENT/DATASET/L_H/BACKBONE/CONFIG.../run_N/seed_N/all_losses.pt

Only current-schema completed manifests are eligible. The command applies the
requested run-selection policy, averages the selected loss tensor, and writes a
complete LaTeX ``table`` environment plus its input manifest.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiment_runs import (
    ManifestError,
    SelectedRun,
    load_manifest,
    manifest_is_selectable,
    select_identity_runs,
    write_report_manifest,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    dataset: str
    setting: str
    method: str
    split: str
    metric: str
    value: float
    path: Path
    seed: int | None = None


def _mean(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu()) if value.numel() else math.nan
    if isinstance(value, Mapping):
        values = [_mean(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        values = [_mean(item) for item in value]
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan
    values = [item for item in values if math.isfinite(item)]
    return sum(values) / len(values) if values else math.nan


def _discover_results_and_runs(
    experiment_dir: str | Path,
    *,
    pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Iterable[str] | None = None,
) -> tuple[list[Result], list[SelectedRun]]:
    root = Path(experiment_dir).expanduser().resolve()
    active_launch = os.environ.get("EXPERIMENT_LAUNCH_ID")
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")
    results: list[Result] = []
    selected_runs: list[SelectedRun] = []
    identity_roots = sorted(
        {path.parent.parent for path in root.rglob("manifest.json") if path.parent.name.startswith("run_") and "archive" not in path.relative_to(root).parts}
    )
    for identity_root in identity_roots:
        manifests = [load_manifest(path) for path in identity_root.glob("run_*/manifest.json")]
        if not any(manifest_is_selectable(manifest, allow_ready_launch_id=active_launch) for manifest in manifests):
            continue
        selected = select_identity_runs(
            identity_root,
            requested_pipeline=pipeline_config,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
            purposes=purposes,
            allow_ready_launch_id=active_launch,
        )
        selected_runs.extend(selected)
        for choice in selected:
            identity = choice.manifest["identity"]
            seed_states = choice.manifest.get("seed_status", {})
            seeds = [int(seed) for seed, state in seed_states.items() if state.get("status") in {"ready", "completed"}]
            paths = [(choice.run_dir / f"seed_{seed}" / "all_losses.pt", seed) for seed in seeds]
            if not paths and (choice.run_dir / "all_losses.pt").is_file():
                paths = [(choice.run_dir / "all_losses.pt", None)]
            for path, seed in paths:
                if not path.is_file():
                    raise ManifestError(f"completed manifest is missing table input: {path}")
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:  # Torch < 2.0
                    payload = torch.load(path, map_location="cpu")
                if not isinstance(payload, Mapping):
                    continue
                for split, split_payload in payload.items():
                    if not isinstance(split_payload, Mapping):
                        continue
                    metrics = {
                        **dict(split_payload.get("losses") or {}),
                        **dict(split_payload.get("summaries") or {}),
                    }
                    for metric, value in metrics.items():
                        results.append(
                            Result(
                                str(identity["dataset"]),
                                f"{identity['lookback']}_{identity['horizon']}",
                                choice.label,
                                str(split),
                                str(metric),
                                _mean(value),
                                path,
                                seed,
                            )
                        )
    return results, selected_runs


def discover_results(experiment_dir: str | Path, **kwargs: Any) -> list[Result]:
    """Load metrics referenced by selected, completed current manifests."""
    return _discover_results_and_runs(experiment_dir, **kwargs)[0]


def _split_names(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = re.split(r"[,;]", value)
    else:
        values = [str(item) for item in value]
    return [item.strip() for item in values if item.strip()]


def _setting_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"[_-]", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _parse_dataset_settings(values: Iterable[str] | None) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"dataset setting must be DATASET=L_H[,L_H], got {item!r}")
        dataset, settings = item.split("=", 1)
        selected.setdefault(dataset.strip(), set()).update(_split_names(settings) or ())
    return selected


def _parse_scale_exponents(values: Iterable[str] | None) -> dict[tuple[str, str], int]:
    exponents: dict[tuple[str, str], int] = {}
    for item in values or ():
        if "=" not in item or "/" not in item.split("=", 1)[0]:
            raise ValueError(f"scale must be DATASET/L_H=EXPONENT, got {item!r}")
        row, exponent = item.split("=", 1)
        dataset, setting = row.split("/", 1)
        exponents[(dataset.strip(), setting.strip())] = int(exponent)
    return exponents


def _latex(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _latex_setting(setting: str) -> str:
    return "--".join(_latex(part) for part in re.split(r"[_-]", setting))


def _auto_exponent(values: Sequence[float], lower_is_better: bool) -> int:
    del lower_is_better
    finite = [abs(value) for value in values if math.isfinite(value) and value != 0]
    if not finite:
        return 0
    finite.sort()
    middle = len(finite) // 2
    anchor = finite[middle] if len(finite) % 2 else (finite[middle - 1] + finite[middle]) / 2.0
    return math.floor(math.log10(anchor))


def _improvement(reference: float, current: float, lower_is_better: bool) -> float:
    if not math.isfinite(reference) or not math.isfinite(current) or reference == 0:
        return math.nan
    sign = 1.0 if lower_is_better else -1.0
    return sign * (reference - current) / abs(reference) * 100.0


def _average_improvements(
    rows: Sequence[Mapping[str, float]], methods: Sequence[str], reference: str, lower_is_better: bool
) -> dict[str, float]:
    output: dict[str, float] = {}
    for method in methods:
        values = [
            _improvement(row.get(reference, math.nan), row.get(method, math.nan), lower_is_better)
            for row in rows
        ]
        values = [value for value in values if math.isfinite(value)]
        output[method] = sum(values) / len(values) if values else math.nan
    return output


def _best(values: Iterable[float], lower_is_better: bool) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return min(finite) if lower_is_better else max(finite)


def _format_cells(
    values: Mapping[str, float], methods: Sequence[str], decimals: int, *,
    lower_is_better: bool, bold: bool, divisor: float = 1.0, percent: bool = False,
    standard_deviations: Mapping[str, float] | None = None,
    show_std: bool = False,
) -> list[str]:
    best = _best(values.values(), lower_is_better)
    cells: list[str] = []
    for method in methods:
        raw = values.get(method, math.nan)
        if not math.isfinite(raw):
            cells.append("--")
            continue
        cell = f"{raw / divisor:.{decimals}f}" + (r"\%" if percent else "")
        deviation = (standard_deviations or {}).get(method, math.nan)
        if show_std and not percent and math.isfinite(deviation):
            cell += rf" $\pm$ {deviation / divisor:.{decimals}f}"
        if bold and best is not None and math.isclose(raw, best, rel_tol=1e-12, abs_tol=1e-15):
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    return cells


def build_table(
    results: Sequence[Result], *, metric: str = "mse", split: str = "test1",
    datasets: Sequence[str] | None = None, settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    methods: Sequence[str] | None = None, reference: str | None = None,
    decimals: int = 2, lower_is_better: bool = True, bold: bool = True,
    dataset_improvements: bool = True, setting_improvements: bool = True,
    overall_improvement: bool = True, auto_scale: bool = True,
    scale_exponent: int | None = None,
    scale_exponents: Mapping[tuple[str, str], int] | None = None,
    show_std: bool = False,
    caption: str | None = None, label: str = "tab:results",
) -> str:
    """Render selected results as a complete LaTeX table environment."""
    wanted_metric, wanted_split = metric.casefold(), split.casefold()
    filtered = [
        result for result in results
        if result.metric.casefold() == wanted_metric and result.split.casefold() == wanted_split
    ]
    available_datasets = sorted({result.dataset for result in filtered}, key=str.casefold)
    dataset_order = list(datasets) if datasets else available_datasets
    dataset_filter = set(dataset_order)
    filtered = [result for result in filtered if result.dataset in dataset_filter]

    global_settings = set(settings or ())
    per_dataset = dataset_settings or {}
    if global_settings or per_dataset:
        filtered = [
            result for result in filtered
            if (
                result.setting in per_dataset[result.dataset]
                if result.dataset in per_dataset
                else not global_settings or result.setting in global_settings
            )
        ]
    available_methods = sorted({result.method for result in filtered}, key=str.casefold)
    if methods:
        method_order = []
        for requested in methods:
            matches = [
                method for method in available_methods
                if method == requested or method.startswith(f"{requested}__") or method.startswith(f"{requested}_run_")
            ]
            method_order.extend(method for method in matches if method not in method_order)
    else:
        method_order = available_methods
    method_filter = set(method_order)
    filtered = [result for result in filtered if result.method in method_filter]
    if not filtered:
        raise ValueError(f"no results match metric={metric!r}, split={split!r}, and the selected filters")
    reference = reference or method_order[0]
    if reference not in method_order:
        raise ValueError(f"reference {reference!r} is not in selected methods {method_order}")

    table: dict[tuple[str, str], dict[str, float]] = {}
    table_std: dict[tuple[str, str], dict[str, float]] = {}
    duplicates: dict[tuple[str, str, str], list[float]] = {}
    for result in filtered:
        duplicates.setdefault((result.dataset, result.setting, result.method), []).append(result.value)
    for (dataset, setting, method), values in duplicates.items():
        finite = [value for value in values if math.isfinite(value)]
        table.setdefault((dataset, setting), {})[method] = (
            sum(finite) / len(finite) if finite else math.nan
        )
        table_std.setdefault((dataset, setting), {})[method] = (
            statistics.stdev(finite) if len(finite) > 1 else math.nan
        )

    dataset_order = [dataset for dataset in dataset_order if any(key[0] == dataset for key in table)]
    settings_by_dataset = {
        dataset: sorted((key[1] for key in table if key[0] == dataset), key=_setting_key)
        for dataset in dataset_order
    }
    observed_settings = sorted({setting for _, setting in table}, key=_setting_key)
    exponent_overrides = scale_exponents or {}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex(caption or f'{metric.upper()} results on {split}.')}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{llc{'r' * len(method_order)}}}",
        r"\toprule",
        "Dataset & $L$--$H$ & Scale & " + " & ".join(_latex(method) for method in method_order) + r" \\",
        r"\midrule",
    ]

    for dataset_index, dataset in enumerate(dataset_order):
        row_settings = settings_by_dataset[dataset]
        for setting_index, setting in enumerate(row_settings):
            row = table[(dataset, setting)]
            if (dataset, setting) in exponent_overrides:
                exponent = exponent_overrides[(dataset, setting)]
            elif scale_exponent is not None:
                exponent = scale_exponent
            elif auto_scale:
                scale_values = [row[reference]] if math.isfinite(row.get(reference, math.nan)) else list(row.values())
                exponent = _auto_exponent(scale_values, lower_is_better)
            else:
                exponent = 0
            dataset_cell = (
                rf"\multirow{{{len(row_settings)}}}{{*}}{{{_latex(dataset)}}}"
                if setting_index == 0 else ""
            )
            cells = _format_cells(
                row, method_order, decimals, lower_is_better=lower_is_better,
                bold=bold, divisor=10.0**exponent,
                standard_deviations=table_std[(dataset, setting)],
                show_std=show_std,
            )
            lines.append(
                " & ".join([dataset_cell, _latex_setting(setting), rf"$\times 10^{{{exponent}}}$", *cells])
                + r" \\"
            )
        if dataset_improvements:
            rows = [table[(dataset, setting)] for setting in row_settings]
            improvements = _average_improvements(rows, method_order, reference, lower_is_better)
            cells = _format_cells(
                improvements, method_order, decimals, lower_is_better=False, bold=bold, percent=True
            )
            lines.append(" & ".join(["", r"\textit{Improvement}", "", *cells]) + r" \\")
        if dataset_index < len(dataset_order) - 1:
            lines.append(r"\midrule")

    if setting_improvements:
        lines.extend([r"\midrule", r"\multicolumn{%d}{l}{\textit{Improvements by setting}} \\" % (3 + len(method_order))])
        for setting in observed_settings:
            rows = [row for (dataset, row_setting), row in table.items() if row_setting == setting]
            improvements = _average_improvements(rows, method_order, reference, lower_is_better)
            cells = _format_cells(
                improvements, method_order, decimals, lower_is_better=False, bold=bold, percent=True
            )
            lines.append(" & ".join(["", _latex_setting(setting), "", *cells]) + r" \\")

    if overall_improvement:
        improvements = _average_improvements(list(table.values()), method_order, reference, lower_is_better)
        cells = _format_cells(
            improvements, method_order, decimals, lower_is_better=False, bold=bold, percent=True
        )
        lines.extend([r"\midrule", " & ".join([r"\multicolumn{2}{l}{Overall improvement}", "", *cells]) + r" \\"])

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", rf"\label{{{_latex(label)}}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def generate_results_table(experiment_dir: str | Path, output: str | Path | None = None, **kwargs: Any) -> Path:
    """Discover losses, render the table, and return the written TeX path."""
    root = Path(experiment_dir).expanduser().resolve()
    default_name = f"results_{str(kwargs.get('metric', 'mse')).lower()}.tex"
    destination = Path(output).expanduser().resolve() if output else root / default_name
    selection_keys = {"pipeline_config", "config_policy", "repeat_policy", "purposes"}
    selection = {key: kwargs.pop(key) for key in list(kwargs) if key in selection_keys}
    results, selected_runs = _discover_results_and_runs(root, **selection)
    latex = build_table(results, **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(latex, encoding="utf-8")
    write_report_manifest(
        destination.parent / "report_manifest.json",
        inputs=selected_runs,
        config_policy=str(selection.get("config_policy", "distinct")),
        repeat_policy=str(selection.get("repeat_policy", "selected")),
        filters={
            "pipeline": dict(selection.get("pipeline_config") or {}),
            "purposes": sorted(selection.get("purposes") or []),
            **{key: kwargs.get(key) for key in ("metric", "split", "datasets", "settings", "methods")},
        },
    )
    return destination


def _value(text: str) -> Any:
    lowered = text.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _pipeline_pairs(values: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"pipeline config must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        output[key] = _value(value)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--split", default="test1")
    parser.add_argument("--datasets", default=None, help="Comma/semicolon-separated subset")
    parser.add_argument("--settings", default=None, help="Global comma/semicolon-separated L_H subset")
    parser.add_argument("--dataset-settings", action="append", default=[], metavar="DATASET=L_H,L_H")
    parser.add_argument("--methods", default=None, help="Comma/semicolon-separated ordered columns")
    parser.add_argument("--reference", default=None, help="Reference method for improvements; default: first column")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--no-dataset-improvements", action="store_true")
    parser.add_argument("--no-setting-improvements", action="store_true")
    parser.add_argument("--no-overall-improvement", action="store_true")
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument("--scale-exponent", type=int, default=None)
    parser.add_argument("--row-scale", action="append", default=[], metavar="DATASET/L_H=EXPONENT")
    parser.add_argument("--show-std", action="store_true", help="Show standard deviation across seeds")
    parser.add_argument("--pipeline-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--config-policy", choices=["distinct", "latest", "average"], default="distinct")
    parser.add_argument("--repeat-policy", choices=["distinct", "latest", "selected", "average"], default="selected")
    parser.add_argument("--purpose", action="append", default=[])
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default="tab:results")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    args = parse_args(argv)
    output = generate_results_table(
        args.experiment_dir,
        args.output,
        metric=args.metric,
        split=args.split,
        datasets=_split_names(args.datasets),
        settings=_split_names(args.settings),
        dataset_settings=_parse_dataset_settings(args.dataset_settings),
        methods=_split_names(args.methods),
        reference=args.reference,
        decimals=args.decimals,
        lower_is_better=not args.higher_is_better,
        bold=not args.no_bold,
        dataset_improvements=not args.no_dataset_improvements,
        setting_improvements=not args.no_setting_improvements,
        overall_improvement=not args.no_overall_improvement,
        auto_scale=not args.no_auto_scale,
        scale_exponent=args.scale_exponent,
        scale_exponents=_parse_scale_exponents(args.row_scale),
        show_std=args.show_std,
        pipeline_config=_pipeline_pairs(args.pipeline_config),
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        purposes=args.purpose,
        caption=args.caption,
        label=args.label,
    )
    LOGGER.info("LaTeX table written to %s", output)
    return output


if __name__ == "__main__":
    main()
