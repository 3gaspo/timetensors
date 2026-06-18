"""PyTorch-only training and evaluation pipeline for TimeTensor models."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import os

import torch
import torch.nn as nn
import torch.optim as optim

from .losses import LossWrapper, build_loss, get_losses
from .models import load_config_dict, resolve_path
from .normalizations import get_normal_stats


TensorTree = torch.Tensor | dict[str, Any] | list[Any] | tuple[Any, ...] | None


@dataclass
class Batch:
    """Normalized batch representation used by ``TorchLearner``."""

    x: torch.Tensor
    y: torch.Tensor
    covariates: Any = None
    past_covariates: torch.Tensor | None = None
    future_covariates: torch.Tensor | None = None
    static_covariates: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LearnerConfig:
    """Small config object for a learner."""

    lr: float = 1e-3
    device: str | torch.device | None = None
    optimizer: str = "adam"
    optimizer_kwargs: Mapping[str, Any] | None = None
    grad_clip: float | None = None
    epochs: int = 1

    @classmethod
    def from_dict(cls, config: Mapping[str, Any] | None) -> "LearnerConfig":
        data = dict(config or {})
        return cls(
            lr=float(data.get("lr", 1e-3)),
            device=data.get("device"),
            optimizer=str(data.get("optimizer", "adam")),
            optimizer_kwargs=data.get("optimizer_kwargs", data.get("kwargs")),
            grad_clip=data.get("grad_clip"),
            epochs=int(data.get("epochs", 1)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LearnerConfig":
        return cls.from_dict(load_config_dict(path))


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed_all(seed)


def cuda_available() -> bool:
    return bool(cuda_diagnostics()["cuda_available"])


def cuda_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnostics["cuda_available"] = torch.cuda.is_available()
        if caught:
            diagnostics["cuda_warnings"] = [str(warning.message) for warning in caught]
    except RuntimeError as exc:
        diagnostics["cuda_available"] = False
        diagnostics["cuda_error"] = str(exc)
    if diagnostics["cuda_available"]:
        diagnostics["cuda_device_count"] = torch.cuda.device_count()
        diagnostics["cuda_current_device"] = torch.cuda.current_device()
        diagnostics["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return diagnostics


def default_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if cuda_available() else "cpu")
    if str(device).lower() in {"gpu", "cuda"}:
        diagnostics = cuda_diagnostics()
        if diagnostics["cuda_available"]:
            return torch.device("cuda")
        raise RuntimeError(f"CUDA was requested but is not available: {diagnostics}")
    return torch.device(device)


def _step_interval(value: Any) -> int | None:
    if value in {None, "None", "none", "null", ""}:
        return None
    return int(value)


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_output_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    return resolve_path(path, base_dir=None if base_dir is None else Path(base_dir))


def move_to_device(value: TensorTree, device: torch.device) -> TensorTree:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def parse_batch(batch: Any) -> Batch:
    """Accept dict batches, ``(x, y)``, ``(x, covariates, y)``, and old tuples."""
    if isinstance(batch, Batch):
        return batch
    if isinstance(batch, Mapping):
        x = _first_present(batch, "x", "X", "input", "inputs", "lookback", "past_values")
        y = _first_present(batch, "y", "Y", "target", "targets", "future_values")
        if x is None or y is None:
            raise ValueError("dict batch requires input and target entries")
        metadata = {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "x",
                "X",
                "input",
                "inputs",
                "lookback",
                "past_values",
                "y",
                "Y",
                "target",
                "targets",
                "future_values",
                "covariates",
                "context",
                "past_covariates",
                "future_covariates",
                "static_covariates",
                "past",
                "future",
                "static",
                "individual_context",
                "global_context",
            }
        }
        covariates = batch.get("covariates", batch.get("context"))
        if covariates is None and (
            batch.get("individual_context") is not None
            or batch.get("global_context") is not None
        ):
            covariates = {
                "individual_context": batch.get("individual_context"),
                "global_context": batch.get("global_context"),
            }
        return Batch(
            x=x,
            y=y,
            covariates=covariates,
            past_covariates=batch.get("past_covariates", batch.get("past")),
            future_covariates=batch.get("future_covariates", batch.get("future")),
            static_covariates=batch.get("static_covariates", batch.get("static")),
            metadata=metadata,
        )
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return Batch(x=batch[0], y=batch[1])
        if len(batch) >= 3:
            metadata = {}
            if len(batch) > 3:
                metadata["extra"] = tuple(batch[3:])
            return Batch(x=batch[0], covariates=batch[1], y=batch[2], metadata=metadata)
    raise TypeError("batch must be a Batch, mapping, or tuple/list")


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def build_optimizer(
    parameters,
    *,
    name: str = "adam",
    lr: float = 1e-3,
    kwargs: Mapping[str, Any] | None = None,
) -> optim.Optimizer:
    kwargs = dict(kwargs or {})
    key = name.lower()
    if key == "adam":
        return optim.Adam(parameters, lr=lr, **kwargs)
    if key == "adamw":
        return optim.AdamW(parameters, lr=lr, **kwargs)
    if key == "sgd":
        return optim.SGD(parameters, lr=lr, **kwargs)
    if key == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, **kwargs)
    raise ValueError(f"unknown optimizer {name!r}")


class TorchLearner:
    """Generic learner for PyTorch forecasting models."""

    def __init__(
        self,
        model: nn.Module,
        criterion: LossWrapper | Mapping[str, Any] | str | None = None,
        *,
        eval_losses: Mapping[str, LossWrapper] | None = None,
        lr: float = 1e-3,
        device: str | torch.device | None = None,
        optimizer: optim.Optimizer | Callable[[Iterable[nn.Parameter]], optim.Optimizer] | None = None,
        optimizer_name: str = "adam",
        optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: Any = None,
        grad_clip: float | None = None,
    ):
        resolved_device = default_device(device)
        self.model = model.to(resolved_device)
        self.device = next(self.model.parameters(), torch.empty(0, device=resolved_device)).device
        self.criterion = build_loss(criterion)
        self.eval_losses = dict(eval_losses or get_losses("mse")[1])
        self.lr = float(lr)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.optimizer_factory = optimizer
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.scheduler_factory = scheduler
        self.optimizer: optim.Optimizer | None = None
        self.scheduler = None
        self.global_step = 0

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: LearnerConfig | Mapping[str, Any] | str | Path | None = None,
        *,
        criterion: LossWrapper | Mapping[str, Any] | str | None = None,
        eval_losses: Mapping[str, LossWrapper] | None = None,
    ) -> "TorchLearner":
        if isinstance(config, LearnerConfig):
            learner_config = config
        elif isinstance(config, (str, Path)):
            learner_config = LearnerConfig.from_yaml(config)
        else:
            learner_config = LearnerConfig.from_dict(config)
        return cls(
            model,
            criterion,
            eval_losses=eval_losses,
            lr=learner_config.lr,
            device=learner_config.device,
            optimizer_name=learner_config.optimizer,
            optimizer_kwargs=learner_config.optimizer_kwargs,
            grad_clip=learner_config.grad_clip,
        )

    def reset_optimizer(self) -> optim.Optimizer:
        if self.optimizer_factory is None:
            self.optimizer = build_optimizer(
                self.model.parameters(),
                name=self.optimizer_name,
                lr=self.lr,
                kwargs=self.optimizer_kwargs,
            )
        elif isinstance(self.optimizer_factory, optim.Optimizer):
            self.optimizer = self.optimizer_factory
        else:
            self.optimizer = self.optimizer_factory(self.model.parameters())
        self.scheduler = None if self.scheduler_factory is None else self.scheduler_factory(self.optimizer)
        return self.optimizer

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "global_step": self.global_step,
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        state = torch.load(Path(path).expanduser().resolve(), map_location=self.device)
        self.model.load_state_dict(state["model"], strict=strict)
        if state.get("optimizer") is not None:
            if self.optimizer is None:
                self.reset_optimizer()
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
        return state

    def train_step(self, batch: Any) -> float:
        if self.optimizer is None:
            self.reset_optimizer()
        assert self.optimizer is not None
        parsed = self._prepare_batch(batch)
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self._predict(parsed)
        mean, std = get_normal_stats(parsed.x, dim=-1, keepdim=True, detach=True)
        loss = self.criterion(prediction, parsed.y, context=parsed.x, mean=mean, std=std)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return float(loss.detach().cpu())

    def train_epoch(self, loader: Iterable[Any]) -> list[float]:
        return [self.train_step(batch) for batch in loader]

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        epochs: int = 1,
        valid_loaders: Mapping[str, Iterable[Any]] | None = None,
        eval_every_steps: int | None = 100,
        log_every_steps: int | None = 1000,
        eval_runs: int = 1,
        seed: int | None = None,
        logger: Any = None,
    ) -> dict[str, Any]:
        set_seed(seed)
        history: dict[str, Any] = {"train": [], "valid": {}}
        start = perf_counter()
        recent_losses: list[float] = []
        run_step = 0
        last_eval_run_step: int | None = None
        steps_per_epoch = _safe_len(train_loader)
        total_steps = None if steps_per_epoch is None else int(epochs) * steps_per_epoch
        eval_interval = _step_interval(eval_every_steps)
        log_interval = _step_interval(log_every_steps)

        if logger is not None:
            logger.info(
                "training epochs=%s steps_per_epoch=%s total_steps=%s eval_every_steps=%s log_every_steps=%s",
                epochs,
                steps_per_epoch if steps_per_epoch is not None else "unknown",
                total_steps if total_steps is not None else "unknown",
                eval_interval,
                log_interval,
            )

        def run_validation(current_run_step: int) -> None:
            if not valid_loaders:
                return
            for name, loader in valid_loaders.items():
                result = self.evaluate(loader, runs=int(eval_runs), seed=None)
                history["valid"].setdefault(name, []).append(
                    {"step": self.global_step, **result}
                )
                if logger is not None:
                    losses = result.get("losses", {})
                    logger.info("step=%s valid[%s]=%s", self.global_step, name, losses)

        for epoch in range(int(epochs)):
            epoch_losses = []
            for batch in train_loader:
                loss = self.train_step(batch)
                run_step += 1
                history["train"].append(loss)
                epoch_losses.append(loss)
                recent_losses.append(loss)
                is_first_step = run_step == 1
                is_final_step = total_steps is not None and run_step == total_steps
                should_log = (
                    logger is not None
                    and (
                        is_first_step
                        or is_final_step
                        or (
                            log_interval is not None
                            and log_interval > 0
                            and run_step % log_interval == 0
                        )
                    )
                )
                if should_log:
                    recent = sum(recent_losses) / max(len(recent_losses), 1)
                    logger.info("step=%s train=%.6f", self.global_step, recent)
                    recent_losses.clear()
                should_eval = bool(
                    valid_loaders
                    and (
                        is_first_step
                        or is_final_step
                        or (
                            eval_interval is not None
                            and eval_interval > 0
                            and run_step % eval_interval == 0
                        )
                    )
                )
                if should_eval:
                    run_validation(run_step)
                    last_eval_run_step = run_step
            if logger is not None and epoch_losses:
                logger.debug(
                    "epoch %s/%s train_loss=%.6f steps=%s",
                    epoch + 1,
                    epochs,
                    sum(epoch_losses) / max(len(epoch_losses), 1),
                    len(epoch_losses),
                )
        if recent_losses and logger is not None:
            recent = sum(recent_losses) / max(len(recent_losses), 1)
            logger.info("step=%s train=%.6f", self.global_step, recent)
        if valid_loaders and last_eval_run_step != run_step:
            run_validation(run_step)
        elapsed_seconds = perf_counter() - start
        history["elapsed_seconds"] = elapsed_seconds
        if logger is not None:
            logger.info("training_done steps=%s seconds=%.2f", run_step, elapsed_seconds)
        return history

    def evaluate(
        self,
        loader: Iterable[Any],
        *,
        return_mode: str = "mean",
        thresholds: Mapping[str, float] | None = None,
        runs: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        set_seed(seed)
        thresholds = dict(thresholds or {})
        losses: dict[str, Any] = {}
        counts: dict[str, int] = {}
        exotics: dict[str, list[dict[str, Any]]] = {}
        self.model.eval()
        with torch.inference_mode():
            for _ in range(int(runs)):
                for raw_batch in loader:
                    batch = self._prepare_batch(raw_batch)
                    prediction = self._predict(batch)
                    mean, std = get_normal_stats(batch.x, dim=-1, keepdim=True, detach=True)
                    for name, criterion in self.eval_losses.items():
                        loss = criterion(
                            prediction,
                            batch.y,
                            context=batch.x,
                            mean=mean,
                            std=std,
                        ).detach()
                        self._collect_loss(
                            losses,
                            counts,
                            exotics,
                            name,
                            loss,
                            batch.metadata,
                            return_mode,
                            thresholds,
                        )
        return {
            "losses": self._finalize_losses(losses, counts, return_mode),
            "exotics": exotics,
        }

    def _prepare_batch(self, batch: Any) -> Batch:
        parsed = parse_batch(batch)
        return Batch(
            x=move_to_device(parsed.x, self.device),
            y=move_to_device(parsed.y, self.device),
            covariates=move_to_device(parsed.covariates, self.device),
            past_covariates=move_to_device(parsed.past_covariates, self.device),
            future_covariates=move_to_device(parsed.future_covariates, self.device),
            static_covariates=move_to_device(parsed.static_covariates, self.device),
            metadata=parsed.metadata,
        )

    def _predict(self, batch: Batch) -> torch.Tensor:
        normalization = getattr(self.model, "normalization", None)
        metadata = batch.metadata or {}
        cluster_ids = metadata.get("cluster_ids")
        model_kwargs = {}
        model_setter = getattr(self.model, "_set_normalization_cluster_ids", None)
        set_cluster_ids = getattr(normalization, "set_cluster_ids", None)
        if callable(model_setter):
            model_kwargs["cluster_ids"] = cluster_ids
        elif callable(set_cluster_ids):
            set_cluster_ids(metadata.get("cluster_ids"))
        return self.model(
            batch.x,
            covariates=batch.covariates,
            past_covariates=batch.past_covariates,
            future_covariates=batch.future_covariates,
            static_covariates=batch.static_covariates,
            **model_kwargs,
        )

    @staticmethod
    def _collect_loss(
        losses: dict[str, Any],
        counts: dict[str, int],
        exotics: dict[str, list[dict[str, Any]]],
        name: str,
        loss: torch.Tensor,
        metadata: Mapping[str, Any] | None,
        return_mode: str,
        thresholds: Mapping[str, float],
    ) -> None:
        sample_loss = loss.mean(dim=tuple(range(1, loss.ndim))) if loss.ndim > 1 else loss
        if name in thresholds:
            high = sample_loss > float(thresholds[name])
            for index in high.nonzero(as_tuple=True)[0].detach().cpu().tolist():
                exotics.setdefault(name, []).append(
                    {
                        "index": int(index),
                        "loss": float(sample_loss[index].detach().cpu()),
                        "metadata": metadata,
                    }
                )
        if return_mode == "mean":
            losses[name] = losses.get(name, 0.0) + float(loss.sum().detach().cpu())
            counts[name] = counts.get(name, 0) + loss.numel()
        elif return_mode == "dim":
            losses.setdefault(name, []).append(loss.mean(dim=0).detach().cpu())
            counts[name] = counts.get(name, 0) + 1
        elif return_mode == "steps":
            losses.setdefault(name, []).extend(sample_loss.detach().cpu().tolist())
        elif return_mode == "all":
            losses.setdefault(name, []).append(loss.detach().cpu())
        else:
            raise ValueError("return_mode must be one of 'mean', 'dim', 'steps', or 'all'")

    @staticmethod
    def _finalize_losses(
        losses: dict[str, Any],
        counts: dict[str, int],
        return_mode: str,
    ) -> dict[str, Any]:
        output = {}
        for name, value in losses.items():
            if return_mode == "mean":
                output[name] = value / max(counts.get(name, 0), 1)
            elif return_mode == "dim":
                output[name] = torch.stack(value, dim=0).mean(dim=0)
            elif return_mode == "steps":
                output[name] = torch.as_tensor(value)
            elif return_mode == "all":
                output[name] = torch.cat(value, dim=0)
        return output


def load_learner(
    model: nn.Module,
    criterion: LossWrapper | Mapping[str, Any] | str | None = None,
    *,
    eval_losses: Mapping[str, LossWrapper] | None = None,
    config: LearnerConfig | Mapping[str, Any] | str | Path | None = None,
    **kwargs,
) -> TorchLearner:
    if config is not None:
        return TorchLearner.from_config(
            model,
            config,
            criterion=criterion,
            eval_losses=eval_losses,
        )
    return TorchLearner(model, criterion, eval_losses=eval_losses, **kwargs)


def train_model(
    learner: TorchLearner,
    loaders: Mapping[str, Iterable[Any]] | Iterable[Any],
    *,
    epochs: int = 1,
    eval_every_steps: int | None = 100,
    log_every_steps: int | None = 1000,
    eval_runs: int = 1,
    seed: int | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    if isinstance(loaders, Mapping):
        train_loader = loaders["train"]
        valid_loaders = {key: value for key, value in loaders.items() if "valid" in key}
    else:
        train_loader = loaders
        valid_loaders = None
    return learner.fit(
        train_loader,
        epochs=epochs,
        valid_loaders=valid_loaders,
        eval_every_steps=eval_every_steps,
        log_every_steps=log_every_steps,
        eval_runs=eval_runs,
        seed=seed,
        logger=logger,
    )
