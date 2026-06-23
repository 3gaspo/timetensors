"""Build publication-ready LaTeX tables from TimeTensor experiment losses.

Expected experiment layout::

    EXPERIMENT/DATASET/L_H/METHOD/all_losses.pt

The command discovers the layout recursively, averages the selected loss tensor,
and writes a complete LaTeX ``table`` environment.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class Result:
    dataset: str
    setting: str
    method: str
    split: str
    metric: str
    value: float
    path: Path


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


def discover_results(experiment_dir: str | Path) -> list[Result]:
    """Load all metrics found below ``experiment_dir``."""
    root = Path(experiment_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")
    results: list[Result] = []
    for path in sorted(root.rglob("all_losses.pt")):
        relative = path.relative_to(root)
        if len(relative.parts) < 4:
            continue
        identity = None
        for index in range(len(relative.parts) - 1, 0, -1):
            if re.fullmatch(r"\d+[_-]\d+", relative.parts[index]):
                identity = (relative.parts[index - 1], relative.parts[index])
                break
        dataset, setting = identity or relative.parts[-4:-2]
        method = path.parent.name
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Torch < 2.0
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            continue
        for split, metrics in payload.items():
            if not isinstance(metrics, Mapping):
                metrics = {"loss": metrics}
            for metric, value in metrics.items():
                results.append(
                    Result(dataset, setting, method, str(split), str(metric), _mean(value), path)
                )
    return results


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
) -> list[str]:
    best = _best(values.values(), lower_is_better)
    cells: list[str] = []
    for method in methods:
        raw = values.get(method, math.nan)
        if not math.isfinite(raw):
            cells.append("--")
            continue
        cell = f"{raw / divisor:.{decimals}f}" + (r"\%" if percent else "")
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
    method_order = list(methods) if methods else available_methods
    method_filter = set(method_order)
    filtered = [result for result in filtered if result.method in method_filter]
    if not filtered:
        raise ValueError(f"no results match metric={metric!r}, split={split!r}, and the selected filters")
    reference = reference or method_order[0]
    if reference not in method_order:
        raise ValueError(f"reference {reference!r} is not in selected methods {method_order}")

    table: dict[tuple[str, str], dict[str, float]] = {}
    duplicates: dict[tuple[str, str, str], list[float]] = {}
    for result in filtered:
        duplicates.setdefault((result.dataset, result.setting, result.method), []).append(result.value)
    for (dataset, setting, method), values in duplicates.items():
        finite = [value for value in values if math.isfinite(value)]
        table.setdefault((dataset, setting), {})[method] = (
            sum(finite) / len(finite) if finite else math.nan
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
    latex = build_table(discover_results(root), **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(latex, encoding="utf-8")
    return destination


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
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default="tab:results")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> Path:
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
        caption=args.caption,
        label=args.label,
    )
    print(f"LaTeX table written to {output}")
    return output


if __name__ == "__main__":
    main()
