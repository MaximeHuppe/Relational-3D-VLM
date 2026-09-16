"""Training loops.

Phase 1: :class:`StageATrainer` pretrains the promptable shape segmenter,
selected on macro-average validation Dice.

Phase 2: :class:`StageBOverfitRunner` drives Stage B onto a single scene until
its training Dice passes ``stage_b_overfit.target_train_dice``. It is a bug
catcher, not a result - channel order, world coordinates, prompt indices and the
decoder all have to be right before a number from Phase 3 means anything.

Phase 3: :class:`StageBTrainer` trains and evaluates Stage B on the full corpus
and is the primary proof-of-concept measurement. Its validation targets are the
two held-out shape classes, so the selection metric is a transfer metric.

Both Stage B loops take an :class:`~src.models.anchor_provider.AnchorProvider`,
which is the single place the ground-truth-versus-predicted anchor choice lives.
Nothing else in the loop changes between the two; the model is handed three mask
channels either way.

Everything that affects a result is explicit: the seed, the device, the
precision, the batch size and the gradient-accumulation factor all come from a
hardware profile in ``configs/train.yaml`` and are written into the checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.evaluation.metrics import (
    PerClassMetrics,
    StratifiedMetrics,
    dice_score,
    format_per_class_table,
    format_stratified_table,
)
from src.training.checkpointing import checkpoint_metadata, save_checkpoint
from src.training.logger import (
    TrainingLogger,
    metrics_from_stage_a,
    metrics_from_stage_b,
)
from src.training.losses import deep_supervision_loss, segmentation_loss


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a device. ``auto`` prefers CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed every generator this project uses."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA here
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingSettings:
    """Resolved training configuration for one stage on one hardware profile."""

    epochs: int = 60
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"
    num_workers: int = 0
    device: str = "auto"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 2
    scheduler: str = "cosine"
    lambda_dice: float = 1.0
    lambda_bce: float = 1.0
    threshold: float = 0.5
    seed: int = 20260915
    model_profile: str = "default"
    hardware_profile: str = "laptop_mps"
    log_every_steps: int = 20
    # Phase 2 only: a step budget instead of an epoch count, and the training
    # Dice the overfit has to reach to count as a pass.
    steps: int = 0
    target_train_dice: float = 0.95
    anchor_source: str = "oracle"

    @classmethod
    def for_stage_a(
        cls,
        *,
        hardware_profile: str = "laptop_mps",
        smoke: bool = False,
        config: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "TrainingSettings":
        """Build from ``configs/train.yaml``; ``smoke=True`` applies the smoke block."""
        train_config = dict(config or load_config("train"))
        common = train_config["common"]
        hardware = train_config["hardware_profiles"][hardware_profile]
        stage = dict(train_config["stage_a"])
        smoke_block = dict(stage.pop("smoke", {}))
        if smoke:
            stage = {**stage, **{k: v for k, v in smoke_block.items() if k != "hardware"}}
            hardware = {**hardware, **smoke_block.get("hardware", {})}

        optimizer = stage["optimizer"]
        scheduler = stage["scheduler"]
        loss = stage["loss"]
        settings = cls(
            epochs=int(stage["epochs"]),
            batch_size=int(hardware["batch_size"]),
            gradient_accumulation_steps=int(hardware["gradient_accumulation_steps"]),
            precision=str(hardware["precision"]),
            num_workers=int(hardware["num_workers"]),
            device=str(hardware.get("device", "auto")),
            learning_rate=float(optimizer["lr"]),
            weight_decay=float(optimizer["weight_decay"]),
            warmup_epochs=int(scheduler.get("warmup_epochs", 0)),
            scheduler=str(scheduler["name"]),
            lambda_dice=float(loss["lambda_dice"]),
            lambda_bce=float(loss["lambda_bce"]),
            threshold=float(stage.get("threshold", 0.5)),
            seed=int(common["seed"]),
            model_profile=str(hardware.get("model_profile", "default")),
            hardware_profile=hardware_profile,
            log_every_steps=int(common.get("log_every_steps", 20)),
        )
        if overrides:
            settings = settings.replace(**overrides)
        return settings

    @classmethod
    def for_stage_b(
        cls,
        *,
        phase: str = "oracle",
        hardware_profile: str = "laptop_mps",
        smoke: bool = False,
        config: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "TrainingSettings":
        """Build Stage B settings from ``configs/train.yaml``.

        Args:
            phase: ``overfit`` (Phase 2, ``stage_b_overfit``) or ``oracle``
                (Phase 3, ``stage_b_oracle``).
            smoke: apply the stage's ``smoke`` block - a few epochs or a few
                hundred steps on the reduced model width.
        """
        if phase not in ("overfit", "oracle"):
            raise ValueError(f"phase must be 'overfit' or 'oracle', got {phase!r}")
        train_config = dict(config or load_config("train"))
        common = train_config["common"]
        hardware = train_config["hardware_profiles"][hardware_profile]
        stage = dict(train_config[f"stage_b_{phase}"])
        smoke_block = dict(stage.pop("smoke", {}))
        if smoke:
            stage = {**stage, **{k: v for k, v in smoke_block.items() if k != "hardware"}}
            hardware = {**hardware, **smoke_block.get("hardware", {})}

        optimizer = stage["optimizer"]
        scheduler = stage["scheduler"]
        loss = stage["loss"]
        settings = cls(
            # Phase 2 counts steps, not epochs; one "epoch" is the whole budget.
            epochs=int(stage.get("epochs", 1)),
            steps=int(stage.get("steps", 0)),
            target_train_dice=float(stage.get("target_train_dice", 0.95)),
            batch_size=int(hardware["batch_size"]),
            gradient_accumulation_steps=int(hardware["gradient_accumulation_steps"]),
            precision=str(hardware["precision"]),
            num_workers=int(hardware["num_workers"]),
            device=str(hardware.get("device", "auto")),
            learning_rate=float(optimizer["lr"]),
            weight_decay=float(optimizer["weight_decay"]),
            warmup_epochs=int(scheduler.get("warmup_epochs", 0)),
            scheduler=str(scheduler["name"]),
            lambda_dice=float(loss["lambda_dice"]),
            lambda_bce=float(loss["lambda_bce"]),
            threshold=float(stage.get("threshold", 0.5)),
            seed=int(common["seed"]),
            model_profile=str(hardware.get("model_profile", "default")),
            hardware_profile=hardware_profile,
            log_every_steps=int(common.get("log_every_steps", 20)),
            anchor_source=str(stage.get("anchor_source", "oracle")),
        )
        if overrides:
            settings = settings.replace(**overrides)
        return settings

    def replace(self, **changes: Any) -> "TrainingSettings":
        from dataclasses import replace as _replace

        return _replace(self, **{k: v for k, v in changes.items() if v is not None})

    PRECISIONS = ("fp32", "bf16", "fp16")

    def __post_init__(self) -> None:
        if self.precision not in self.PRECISIONS:
            raise ValueError(
                f"precision must be one of {self.PRECISIONS}, got {self.precision!r}"
            )

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """The autocast dtype, or ``None`` to run in full precision."""
        return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)

    def autocast_dtype_on(self, device: torch.device) -> torch.dtype | None:
        """The autocast dtype to actually use on ``device``.

        Autocast is disabled on CPU whatever the profile says. Measured on this
        project's Stage B model, one forward+backward at batch 2 on 64^3:
        **0.93 s in float32 against 30.75 s under bf16 autocast** - a 33x
        regression, because PyTorch has no fused bf16 path for `conv3d` on CPU
        and falls back to casting around every kernel. The profiles pick bf16
        for MPS and CUDA, where it is a speedup; a `--device cpu` run (tests,
        debugging, a machine without an accelerator) would otherwise inherit it
        and look hung.
        """
        dtype = self.autocast_dtype
        if dtype is None or device.type == "cpu":
            return None
        return dtype

    @property
    def needs_grad_scaler(self) -> bool:
        """Only float16 needs loss scaling.

        Measured on this project's Stage A model, one backward pass at batch 2:
        float16 leaves 3.2% of gradient elements exactly zero against 0.5% for
        float32, because float16's dynamic range cannot represent gradients of a
        loss dominated by 262,144 background voxels. bfloat16 has float32's
        exponent range, matches its gradient norm to three digits, and needs no
        scaler - which is why it is the default autocast dtype here.
        """
        return self.precision == "fp16"


def build_optimizer(model: nn.Module, settings: TrainingSettings) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, settings: TrainingSettings
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine decay, stepped once per epoch."""
    warmup = max(int(settings.warmup_epochs), 0)
    total = max(int(settings.epochs), 1)

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if settings.scheduler != "cosine":
            return 1.0
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _epoch_progress(
    loader: DataLoader,
    *,
    epoch: int,
    epochs: int,
    verbose: bool,
) -> DataLoader | tqdm:
    """In-place tqdm over one epoch's batches; the loader itself when quiet."""
    if not verbose:
        return loader
    return tqdm(
        loader,
        desc=f"epoch {epoch}/{max(epochs - 1, 0)}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.3,
    )


def _update_progress(progress: DataLoader | tqdm, **stats: object) -> None:
    if isinstance(progress, tqdm):
        progress.set_postfix(stats, refresh=True)


def _training_logger(
    output_dir: Path,
    settings: TrainingSettings,
    *,
    log_cfg: Mapping[str, Any] | None,
    run_name: str | None,
    full_config: Mapping[str, Any] | None,
    verbose: bool,
    total_epochs: int | None = None,
) -> TrainingLogger:
    """JSONL + stdout + optional W&B logger for one training run."""
    return TrainingLogger(
        output_dir,
        total_epochs=total_epochs if total_epochs is not None else settings.epochs,
        log_cfg=log_cfg or {},
        run_name=run_name or output_dir.name,
        full_config=full_config,
        verbose=verbose,
    )


@dataclass
class EpochResult:
    """One epoch of training plus its validation metrics."""

    epoch: int
    train_loss: float
    train_components: dict[str, float]
    val_metrics: dict[str, Any]
    learning_rate: float
    seconds: float
    train_dice: float = 0.0

    @property
    def val_dice(self) -> float:
        return float(self.val_metrics.get("mean", {}).get("dice", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_dice": self.train_dice,
            "train_components": self.train_components,
            "val": self.val_metrics,
            "learning_rate": self.learning_rate,
            "seconds": self.seconds,
        }


class StageATrainer:
    """Trains the promptable shape segmenter and reports per-class Dice/IoU."""

    def __init__(
        self,
        model: nn.Module,
        settings: TrainingSettings,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        *,
        class_names: Sequence[str],
        output_dir: Path | str,
        verbose: bool = True,
        log_cfg: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        full_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = tuple(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.log_cfg = dict(log_cfg or {})
        self.run_name = run_name
        self.full_config = dict(full_config) if full_config is not None else None
        self.optimizer = build_optimizer(self.model, settings)
        self.scheduler = build_scheduler(self.optimizer, settings)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=settings.needs_grad_scaler
        )
        self.history: list[EpochResult] = []
        self.best_dice = -1.0
        self.best_epoch = -1

    # -- internals ---------------------------------------------------------
    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _autocast(self):
        dtype = self.settings.autocast_dtype_on(self.device)
        if dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _deep_supervision_weights(self) -> list[float]:
        weights = getattr(self.model, "config", None)
        if weights is None:
            return [1.0]
        return list(weights.deep_supervision_weights)

    # -- training ----------------------------------------------------------
    def train_epoch(self, epoch: int) -> tuple[float, dict[str, float], float]:
        self.model.train()
        accumulation = max(self.settings.gradient_accumulation_steps, 1)
        totals: dict[str, float] = {}
        total_loss, total_dice, steps = 0.0, 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = _epoch_progress(
            self.train_loader,
            epoch=epoch,
            epochs=self.settings.epochs,
            verbose=self.verbose,
        )
        for index, batch in enumerate(progress):
            batch = self._to_device(batch)
            with self._autocast():
                output = self.model(batch["scene_volume"], batch["prompt_ids"])
                loss, components = deep_supervision_loss(
                    output.deep_supervision,
                    batch["target_masks"],
                    self._deep_supervision_weights(),
                    lambda_dice=self.settings.lambda_dice,
                    lambda_bce=self.settings.lambda_bce,
                )
            self.scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                total_dice += float(
                    dice_score(
                        output.logits.detach().float(),
                        batch["target_masks"],
                        threshold=self.settings.threshold,
                        from_logits=True,
                    ).mean()
                )
            total_loss += float(loss.detach())
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
            _update_progress(
                progress,
                loss=f"{total_loss / steps:.4f}",
                train_dice=f"{total_dice / steps:.4f}",
            )

        if steps % accumulation != 0:
            # Flush a partial accumulation window rather than dropping it.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        divisor = max(steps, 1)
        return (
            total_loss / divisor,
            {key: value / divisor for key, value in totals.items()},
            total_dice / divisor,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict[str, Any]:
        """Per-class Dice and IoU at full resolution on the inference map."""
        loader = loader or self.val_loader
        if loader is None:
            return {}
        self.model.eval()
        metrics = PerClassMetrics(self.class_names)
        total_loss, steps = 0.0, 0
        for batch in loader:
            batch = self._to_device(batch)
            with self._autocast():
                output = self.model(batch["scene_volume"], batch["prompt_ids"], deep_supervision=False)
            loss, _ = segmentation_loss(
                output.logits.float(),
                batch["target_masks"],
                lambda_dice=self.settings.lambda_dice,
                lambda_bce=self.settings.lambda_bce,
            )
            total_loss += float(loss)
            steps += 1
            metrics.update(
                output.logits.float().cpu(),
                batch["target_masks"].cpu(),
                class_ids=batch["prompt_ids"].cpu(),
                threshold=self.settings.threshold,
                from_logits=True,
            )
        summary = metrics.summary()
        summary["loss"] = total_loss / max(steps, 1)
        summary["table"] = format_per_class_table(metrics, self.class_names)
        return summary

    def fit(self) -> list[EpochResult]:
        """Run the full schedule, checkpointing the best macro-average Dice."""
        seed_everything(self.settings.seed)
        if self.verbose:
            scaler_note = " + grad scaler" if self.settings.needs_grad_scaler else ""
            print(
                f"device: {self.device} | precision: {self.settings.precision}{scaler_note} | "
                f"batch {self.settings.batch_size} x {self.settings.gradient_accumulation_steps} "
                f"accumulation"
            )
            print(
                f"model: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M parameters "
                f"({self.settings.model_profile} profile)"
            )
        with _training_logger(
            self.output_dir,
            self.settings,
            log_cfg=self.log_cfg,
            run_name=self.run_name,
            full_config=self.full_config,
            verbose=self.verbose,
        ) as logger:
            for epoch in range(self.settings.epochs):
                started = time.perf_counter()
                train_loss, components, train_dice = self.train_epoch(epoch)
                self.scheduler.step()
                val_metrics = self.evaluate()
                result = EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_dice=train_dice,
                    train_components=components,
                    val_metrics=val_metrics,
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    seconds=time.perf_counter() - started,
                )
                self.history.append(result)
                logger.log_epoch(
                    epoch + 1,
                    metrics_from_stage_a(
                        train_loss=train_loss,
                        train_components=components,
                        train_dice=train_dice,
                        val_metrics=val_metrics,
                    ),
                    result.learning_rate,
                )
                if result.val_dice > self.best_dice:
                    self.best_dice = result.val_dice
                    self.best_epoch = epoch
                    self.save("best.pt", result)
            logger.print_summary()
        self.save("last.pt", self.history[-1] if self.history else None)
        self.write_history()
        return self.history

    # -- artefacts ---------------------------------------------------------
    def save(self, name: str, result: EpochResult | None) -> Path:
        metadata = checkpoint_metadata(
            stage="stage_a",
            epoch=result.epoch if result else -1,
            metrics={k: v for k, v in (result.val_metrics if result else {}).items() if k != "table"},
            settings=self.settings,
            seed=self.settings.seed,
            selection_metric="val_mean_dice",
            extra={
                "best_dice": self.best_dice,
                "best_epoch": self.best_epoch,
                "class_names": list(self.class_names),
            },
        )
        return save_checkpoint(
            self.output_dir / name,
            model=self.model,
            metadata=metadata,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            model_config=getattr(self.model, "config", None),
        )

    def write_history(self) -> Path:
        path = self.output_dir / "history.json"
        path.write_text(
            json.dumps([result.to_dict() for result in self.history], indent=2) + "\n",
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Stage B (Phases 2-4)
# ---------------------------------------------------------------------------
@dataclass
class StageBEpochResult:
    """One Stage B epoch: training loss plus the validation report."""

    epoch: int
    train_loss: float
    train_components: dict[str, float]
    train_dice: float
    val_metrics: dict[str, Any]
    learning_rate: float
    seconds: float

    @property
    def val_dice(self) -> float:
        return float(self.val_metrics.get("overall", {}).get("dice", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_components": self.train_components,
            "train_dice": self.train_dice,
            "val": {k: v for k, v in self.val_metrics.items() if k != "table"},
            "learning_rate": self.learning_rate,
            "seconds": self.seconds,
        }


class StageBTrainer:
    """Trains the relational target segmenter (Phase 3) and reports Stage B metrics.

    The loop itself is deliberately dull; the two things that matter are:

    * every model call goes through :func:`src.data.dataset.stage_b_model_inputs`,
      so the target mask cannot reach the network even by accident;
    * the anchor channels come from an
      :class:`~src.models.anchor_provider.AnchorProvider`, so training or
      evaluating on Stage A's predicted anchors instead of the ground-truth ones
      is a constructor argument, not a code change.

    Args:
        model: a :class:`~src.models.relational_vlm.RelationalVLM`.
        settings: from :meth:`TrainingSettings.for_stage_b`.
        anchor_provider: ``oracle`` or ``predicted``; defaults to oracle.
        stage: the name recorded in the checkpoint (``stage_b_oracle`` etc.).
    """

    def __init__(
        self,
        model: nn.Module,
        settings: TrainingSettings,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        *,
        output_dir: Path | str,
        anchor_provider: Any | None = None,
        stage: str = "stage_b_oracle",
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        verbose: bool = True,
        log_cfg: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        full_config: Mapping[str, Any] | None = None,
    ) -> None:
        from src.models.anchor_provider import OracleAnchorProvider

        self.settings = settings
        self.device = resolve_device(settings.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.anchor_provider = anchor_provider or OracleAnchorProvider()
        self.stage = stage
        self.spacing = tuple(float(v) for v in spacing)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.log_cfg = dict(log_cfg or {})
        self.run_name = run_name
        self.full_config = dict(full_config) if full_config is not None else None
        self.optimizer = build_optimizer(self.model, settings)
        self.scheduler = build_scheduler(self.optimizer, settings)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=settings.needs_grad_scaler
        )
        self.history: list[StageBEpochResult] = []
        self.best_dice = -1.0
        self.best_epoch = -1

    # -- internals ---------------------------------------------------------
    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _autocast(self):
        dtype = self.settings.autocast_dtype_on(self.device)
        if dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _forward(self, batch: Mapping[str, Any]):
        """Anchors from the provider, then the model - and nothing else."""
        from src.data.dataset import stage_b_model_inputs

        anchor_masks = self.anchor_provider(batch).to(self.device)
        return self.model(**stage_b_model_inputs(batch, anchor_masks))

    def _loss(self, logits: Tensor, batch: Mapping[str, Any]):
        return segmentation_loss(
            logits.float(),
            batch["target_mask"],
            lambda_dice=self.settings.lambda_dice,
            lambda_bce=self.settings.lambda_bce,
        )

    # -- training ----------------------------------------------------------
    def train_epoch(self, epoch: int) -> tuple[float, dict[str, float], float]:
        """One pass over the training examples. Returns loss, components, Dice."""
        self.model.train()
        accumulation = max(self.settings.gradient_accumulation_steps, 1)
        totals: dict[str, float] = {}
        total_loss, total_dice, steps = 0.0, 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = _epoch_progress(
            self.train_loader,
            epoch=epoch,
            epochs=self.settings.epochs,
            verbose=self.verbose,
        )
        for index, raw in enumerate(progress):
            batch = self._to_device(raw)
            with self._autocast():
                output = self._forward(batch)
            loss, components = self._loss(output.logits, batch)
            self.scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                total_dice += float(
                    dice_score(
                        output.logits.detach().float(),
                        batch["target_mask"],
                        threshold=self.settings.threshold,
                        from_logits=True,
                    ).mean()
                )
            total_loss += float(loss.detach())
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
            _update_progress(
                progress,
                loss=f"{total_loss / steps:.4f}",
                train_dice=f"{total_dice / steps:.4f}",
            )

        if steps % accumulation != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        divisor = max(steps, 1)
        return (
            total_loss / divisor,
            {key: value / divisor for key, value in totals.items()},
            total_dice / divisor,
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader | None = None,
        *,
        with_hausdorff: bool = False,
        percentile: float | None = None,
    ) -> dict[str, Any]:
        """Dice, IoU and (optionally) Hausdorff, aggregate and stratified.

        Hausdorff is off by default because it is the only metric here that
        costs a surface extraction and a pairwise distance per sample; the final
        report turns it on.
        """
        loader = loader or self.val_loader
        if loader is None:
            return {}
        self.model.eval()
        # Anchor statistics describe *this* evaluation, not everything the
        # provider has seen since the run started.
        reset = getattr(self.anchor_provider, "reset", None)
        if callable(reset):
            reset()
        metrics = StratifiedMetrics()
        total_loss, steps = 0.0, 0
        for raw in loader:
            batch = self._to_device(raw)
            with self._autocast():
                output = self._forward(batch)
            loss, _ = self._loss(output.logits, batch)
            total_loss += float(loss)
            steps += 1
            metrics.update(
                output.logits.float().cpu(),
                batch["target_mask"].cpu(),
                target_shapes=batch["target_shape_name"],
                anchor_shapes=batch["anchor_shape_names"],
                directions=batch["directions"],
                threshold=self.settings.threshold,
                from_logits=True,
                with_hausdorff=with_hausdorff,
                percentile=percentile,
                spacing=self.spacing,
            )
        summary = metrics.summary()
        summary["loss"] = total_loss / max(steps, 1)
        summary["anchor_source"] = getattr(self.anchor_provider, "source", "oracle")
        quality = getattr(self.anchor_provider, "anchor_quality", None)
        if callable(quality):
            measured = quality()
            if measured:
                summary["anchor_quality"] = measured
        summary["table"] = format_stratified_table(summary)
        return summary

    def fit(self) -> list[StageBEpochResult]:
        """Run the schedule, checkpointing the best validation Dice."""
        seed_everything(self.settings.seed)
        if self.verbose:
            self.print_header()
        with _training_logger(
            self.output_dir,
            self.settings,
            log_cfg=self.log_cfg,
            run_name=self.run_name,
            full_config=self.full_config,
            verbose=self.verbose,
        ) as logger:
            for epoch in range(self.settings.epochs):
                started = time.perf_counter()
                train_loss, components, train_dice = self.train_epoch(epoch)
                self.scheduler.step()
                val_metrics = self.evaluate()
                result = StageBEpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_components=components,
                    train_dice=train_dice,
                    val_metrics=val_metrics,
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    seconds=time.perf_counter() - started,
                )
                self.history.append(result)
                logger.log_epoch(
                    epoch + 1,
                    metrics_from_stage_b(
                        train_loss=train_loss,
                        train_components=components,
                        train_dice=train_dice,
                        val_metrics=val_metrics,
                    ),
                    result.learning_rate,
                )
                if result.val_dice > self.best_dice:
                    self.best_dice = result.val_dice
                    self.best_epoch = epoch
                    self.save("best.pt", result)
            logger.print_summary()
        self.save("last.pt", self.history[-1] if self.history else None)
        self.write_history()
        return self.history

    def print_header(self) -> None:
        print(
            f"device: {self.device} | precision: {self.effective_precision()} | "
            f"batch {self.settings.batch_size} x {self.settings.gradient_accumulation_steps} "
            f"accumulation"
        )
        print(
            f"model: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M parameters "
            f"({self.settings.model_profile} profile, variant "
            f"{getattr(getattr(self.model, 'config', None), 'variant', 'full')})"
        )
        print(f"anchors: {getattr(self.anchor_provider, 'source', 'oracle')}")

    def effective_precision(self) -> str:
        """What the loop actually runs in - autocast is dropped on CPU."""
        dtype = self.settings.autocast_dtype_on(self.device)
        if dtype is None and self.settings.precision != "fp32":
            return f"fp32 ({self.settings.precision} autocast unavailable on {self.device.type})"
        suffix = " + grad scaler" if self.settings.needs_grad_scaler else ""
        return f"{self.settings.precision}{suffix}"

    # -- artefacts ---------------------------------------------------------
    def save(self, name: str, result: StageBEpochResult | None) -> Path:
        provider_summary = getattr(self.anchor_provider, "summary", None)
        metadata = checkpoint_metadata(
            stage=self.stage,
            epoch=result.epoch if result else -1,
            metrics={
                k: v for k, v in (result.val_metrics if result else {}).items() if k != "table"
            },
            settings=self.settings,
            seed=self.settings.seed,
            selection_metric="val_dice",
            extra={
                "best_dice": self.best_dice,
                "best_epoch": self.best_epoch,
                "anchors": provider_summary() if callable(provider_summary) else {},
            },
        )
        return save_checkpoint(
            self.output_dir / name,
            model=self.model,
            metadata=metadata,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            model_config=getattr(self.model, "config", None),
        )

    def write_history(self) -> Path:
        path = self.output_dir / "history.json"
        path.write_text(
            json.dumps([result.to_dict() for result in self.history], indent=2) + "\n",
            encoding="utf-8",
        )
        return path


class StageBOverfitRunner(StageBTrainer):
    """Phase 2: drive Stage B onto one scene until it (nearly) memorises it.

    A step budget rather than an epoch count, because the training set is a
    handful of examples: the loader is cycled until ``settings.steps`` optimiser
    steps have run, or until the running training Dice passes
    ``settings.target_train_dice`` over a full pass.

    What a pass proves: the anchor channels line up with the prompt clauses, the
    world coordinates are oriented as the direction rules assume, the intersection
    module can express "all three at once", and the decoder can put a mask where
    the evidence is. What it does not prove: anything about generalisation. That
    is Phase 3's job.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stage", "stage_b_overfit")
        super().__init__(*args, **kwargs)
        self.steps_run = 0
        self.reached_target = False
        self.best_train_dice = 0.0
        self.trace: list[dict[str, float]] = []

    def run(self) -> dict[str, Any]:
        """Run the overfit and return its report."""
        seed_everything(self.settings.seed)
        if self.verbose:
            self.print_header()
            print(
                f"budget: {self.settings.steps} steps, target train dice "
                f"{self.settings.target_train_dice}"
            )
        budget = max(int(self.settings.steps), 1)
        accumulation = max(self.settings.gradient_accumulation_steps, 1)
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        with _training_logger(
            self.output_dir,
            self.settings,
            log_cfg=self.log_cfg,
            run_name=self.run_name,
            full_config=self.full_config,
            verbose=False,
            total_epochs=budget,
        ) as logger:
            pass_index = 0
            while self.steps_run < budget and not self.reached_target:
                losses: list[float] = []
                dices: list[float] = []
                totals: dict[str, float] = {}
                for index, raw in enumerate(self.train_loader):
                    if self.steps_run >= budget:
                        break
                    batch = self._to_device(raw)
                    with self._autocast():
                        output = self._forward(batch)
                    loss, components = self._loss(output.logits, batch)
                    self.scaler.scale(loss / accumulation).backward()
                    if (index + 1) % accumulation == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.steps_run += 1
                    losses.append(float(loss.detach()))
                    for key, value in components.items():
                        totals[key] = totals.get(key, 0.0) + value
                    with torch.no_grad():
                        dices.append(
                            float(
                                dice_score(
                                    output.logits.detach().float(),
                                    batch["target_mask"],
                                    threshold=self.settings.threshold,
                                    from_logits=True,
                                ).mean()
                            )
                        )

                n_steps = max(len(losses), 1)
                mean_loss = sum(losses) / n_steps
                mean_dice = sum(dices) / n_steps
                mean_components = {key: value / n_steps for key, value in totals.items()}
                self.best_train_dice = max(self.best_train_dice, mean_dice)
                self.trace.append(
                    {"pass": pass_index, "steps": self.steps_run, "loss": mean_loss, "dice": mean_dice}
                )
                pass_metrics = metrics_from_stage_b(
                    train_loss=mean_loss,
                    train_components=mean_components,
                    train_dice=mean_dice,
                )
                if self.verbose and (pass_index % 10 == 0 or mean_dice >= self.settings.target_train_dice):
                    print(
                        f"  pass {pass_index:>4}  step {self.steps_run:>5}/{budget}  "
                        f"loss {mean_loss:7.4f}  train dice {mean_dice:6.4f}"
                    )
                if mean_dice >= self.settings.target_train_dice:
                    # The running mean is measured while the model is still moving,
                    # so it can cross the target before a clean pass does. Confirm
                    # on an eval pass before declaring success, otherwise keep going.
                    clean_metrics = self.evaluate(self.train_loader)
                    clean = float(clean_metrics["overall"]["dice"])
                    self.best_train_dice = max(self.best_train_dice, clean)
                    pass_metrics["train_dice_clean"] = clean
                    pass_metrics["val_dice"] = clean
                    pass_metrics["val_loss"] = float(clean_metrics.get("loss", 0.0))
                    if self.verbose:
                        print(f"         clean train dice {clean:6.4f}")
                    self.reached_target = clean >= self.settings.target_train_dice
                    self.model.train()
                logger.log_epoch(self.steps_run, pass_metrics, learning_rate)
                pass_index += 1

            report = self.report(time.perf_counter() - started)
            logger.log_epoch(
                max(self.steps_run, 1),
                metrics_from_stage_b(
                    train_loss=self.trace[-1]["loss"] if self.trace else 0.0,
                    train_components={},
                    train_dice=float(report["final_train_dice"]),
                    val_metrics=report.get("metrics"),
                ),
                learning_rate,
            )

        self.save("last.pt", None)
        (self.output_dir / "overfit_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report

    @torch.no_grad()
    def final_metrics(self) -> dict[str, Any]:
        """Per-example Dice on the overfit set, with Hausdorff."""
        return self.evaluate(self.train_loader, with_hausdorff=True)

    def report(self, seconds: float) -> dict[str, Any]:
        metrics = self.final_metrics()
        return {
            "stage": self.stage,
            "passed": bool(self.reached_target),
            "target_train_dice": self.settings.target_train_dice,
            "best_train_dice": self.best_train_dice,
            "final_train_dice": float(metrics.get("overall", {}).get("dice", 0.0)),
            "steps_run": self.steps_run,
            "step_budget": int(self.settings.steps),
            "seconds": seconds,
            "anchor_source": getattr(self.anchor_provider, "source", "oracle"),
            "examples": int(len(self.train_loader.dataset)),  # type: ignore[arg-type]
            "metrics": {k: v for k, v in metrics.items() if k != "table"},
            "trace": self.trace,
        }
