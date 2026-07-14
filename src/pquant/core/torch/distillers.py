from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from pquant.core.torch.layers import (
    PQWeightBiasBase,
    get_model_losses,
    post_epoch_functions,
    post_pretrain_functions,
    post_round_functions,
    pre_epoch_functions,
    pre_finetune_functions,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def get_module(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise ValueError(f"Module '{name}' not found in model")


def pq_layer_names(model: nn.Module) -> list[str]:
    """Return names of all PQWeightBiasBase submodules in forward order."""
    return [name for name, m in model.named_modules() if isinstance(m, PQWeightBiasBase)]


class CachedDataset(torch.utils.data.Dataset):
    """Per-batch cache backed by files in a temporary directory.

    Each file stores one ``(input, output)`` batch as a tuple.
    Since each file has one batch, uses batch_size of 1 here.
    """

    def __init__(self, cache_dir: str | Path, n_batches: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_batches = n_batches

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.load(
            self.cache_dir / f"{idx:08d}.pt",
            weights_only=True,
        )


class LayerwiseDistiller:
    """
    Distills a teacher model into a student model one PQ layer at a time.

    For each layer, all student parameters are frozen except that layer's,
    and the layer is trained to match the teacher's outputs at that point.

    Args:
        teacher: Reference model. Will be set to eval() and
                 gradients disabled during distillation.
        student: PQuantML model.
        loss_fn: Loss function between student and teacher activations.
        precompute_layer_inputs: If True, run one pass over the dataloader at
                       the start of ``distill_layer``, capturing the teacher
                       layer's input and output. Pairs
                       are saved to a temporary directory on disk. All epoch loops
                       then call ``student_layer(layer_input)`` directly, skipping the full model forward entirely.
        prefetch_workers: Number of DataLoader worker processes used to
                       prefetch cached batches from disk when
                       ``precompute_layer_inputs=True``.
        cache_dir:     Directory under which temporary per-layer cache
                       subdirectories are created when ``precompute_layer_inputs=True``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: torch.device | None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        precompute_layer_inputs: bool = True,
        prefetch_workers: int = 2,
        cache_dir: str | None = None,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.loss_fn = loss_fn or F.mse_loss
        self.precompute_layer_inputs = precompute_layer_inputs
        self.prefetch_workers = prefetch_workers
        self.cache_dir = cache_dir

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def precompute_layer_io(
        self,
        teacher_layer: nn.Module,
        dataloader: Iterable,
        cache_dir: str,
    ) -> torch.utils.data.DataLoader:
        """Cache ``(layer_input, layer_output)``.

        Runs one teacher pass and one frozen student pass per batch, saving
        each pair. Returns a DataLoader which uses this saved data.
        """
        teacher_captured_data: dict[str, torch.Tensor] = {}

        def teacher_post(m: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            teacher_output = out[0] if isinstance(out, tuple) else out
            teacher_captured_data["out"] = teacher_output.detach().cpu()

        def teacher_pre(m: nn.Module, inp: tuple) -> None:
            teacher_captured_data["inp"] = inp[0].detach().cpu()

        teacher_output_hook = teacher_layer.register_forward_hook(teacher_post)
        teacher_input_hook = teacher_layer.register_forward_pre_hook(teacher_pre)

        n_batches = 0
        try:
            with torch.no_grad():
                for x, _ in dataloader:
                    if self.device is not None:
                        x = x.to(self.device)
                    self.teacher(x)
                    torch.save(
                        (teacher_captured_data["inp"], teacher_captured_data["out"]),
                        Path(cache_dir) / f"{n_batches:08d}.pt",
                    )
                    n_batches += 1
        finally:
            teacher_output_hook.remove()
            teacher_input_hook.remove()

        dataset = CachedDataset(cache_dir, n_batches)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.prefetch_workers,
            collate_fn=lambda b: b[0],
        )

    def val_layer_loss(
        self,
        teacher_layer: nn.Module,
        student_layer: nn.Module,
        val_dataloader: Iterable,
    ) -> float:
        """Compute mean validation loss for a single layer (no grad, eval mode).

        Always runs a full model forward pass with hooks — the val dataset is
        never cached to disk, avoiding large intermediate feature map storage.
        """
        t_captured: dict[str, torch.Tensor] = {}
        s_captured: dict[str, torch.Tensor] = {}

        def teacher_post(m: nn.Module, inp: tuple, out) -> None:
            t_captured["out"] = (out[0] if isinstance(out, tuple) else out).detach()

        def student_hook(m: nn.Module, inp: tuple, out) -> None:
            s_captured["out"] = out[0] if isinstance(out, tuple) else out

        h_t = teacher_layer.register_forward_hook(teacher_post)
        h_s = student_layer.register_forward_hook(student_hook)

        self.student.eval()
        batch_losses: list[float] = []
        try:
            with torch.no_grad():
                for x, _ in val_dataloader:
                    if self.device is not None:
                        x = x.to(self.device)
                    self.teacher(x)
                    self.student(x)
                    batch_losses.append(self.loss_fn(s_captured["out"], t_captured["out"]).item())
        finally:
            h_t.remove()
            h_s.remove()
            self.student.train()

        return sum(batch_losses) / len(batch_losses)

    def distill_layer(
        self,
        teacher_layer_name: str,
        student_layer_name: str,
        dataloader: Iterable,
        optimizer_factory: Callable[[Iterable], torch.optim.Optimizer],
        n_epochs: int,
        val_dataloader: Iterable | None = None,
        epoch_callback: Callable[[int, float, float | None, str], None] | None = None,
    ) -> list[float]:
        """
        Distill a single layer.

        Args:
            teacher_layer_name: Named path to the module in the teacher, e.g. ``"layers.0"``.
            student_layer_name: Named path to the corresponding module in the student.
            dataloader:  Iterable of ``(inputs, targets)`` batches.
            optimizer_factory: Callable that accepts an iterable of parameters and
                         returns a fresh optimizer. Called once per layer with only
                         the target layer's parameters, so state never carries over
                         between layers. Example:
                         ``lambda params: torch.optim.Adam(params, lr=1e-4)``.
            n_epochs:    Number of full passes through the dataloader.
                         ``post_epoch_functions`` is called on the layer after
                         each pass.
            val_dataloader: Optional validation dataloader. When provided, a
                         no-grad eval pass is run after every training epoch and
                         the resulting mean loss is passed to ``epoch_callback``.
            epoch_callback: Called after each epoch with
                         ``(epoch, train_loss, val_loss, student_layer_name)``
                         where ``val_loss`` is ``None`` when no ``val_dataloader``
                         is given. Use this to drive per-epoch side effects such
                         as calling ``student.increment_alpha()``, stepping a
                         scheduler, or logging metrics.

        Returns:
            List of per-epoch mean training losses.
        """
        teacher_layer = get_module(self.teacher, teacher_layer_name)
        student_layer = get_module(self.student, student_layer_name)

        # Freeze everything except the target layer (skip non-float params
        # like bool pruning masks which cannot require gradients)
        frozen: dict[str, bool] = {}
        for name, param in self.student.named_parameters():
            if not param.is_floating_point():
                continue
            frozen[name] = param.requires_grad
            param.requires_grad_(name.startswith(student_layer_name))

        optimizer = optimizer_factory(student_layer.parameters())

        tmpdir = None
        hooks = []
        epoch_losses: list[float] = []
        self.student.train()

        try:
            if self.precompute_layer_inputs:
                tmpdir = tempfile.TemporaryDirectory(prefix="ldistil_", dir=self.cache_dir or str(Path.cwd()))
                dataloader = self.precompute_layer_io(teacher_layer, dataloader, tmpdir.name)
            else:
                teacher_out: dict[str, torch.Tensor] = {}
                student_out: dict[str, torch.Tensor] = {}

                def _teacher_hook(m: nn.Module, inp: tuple, out) -> None:
                    t = out[0] if isinstance(out, tuple) else out
                    teacher_out["out"] = t.detach()

                def student_hook(m: nn.Module, inp: tuple, out) -> None:
                    t = out[0] if isinstance(out, tuple) else out
                    student_out["out"] = t

                h_t = teacher_layer.register_forward_hook(_teacher_hook)
                h_s = student_layer.register_forward_hook(student_hook)
                hooks = [h_t, h_s]

            for epoch in range(n_epochs):
                if getattr(student_layer, "enable_pruning", False):
                    student_layer.pruning_layer.pre_epoch_function(epoch, n_epochs)
                batch_losses: list[float] = []

                if self.precompute_layer_inputs:
                    for t_in, t_out in dataloader:
                        if self.device is not None:
                            t_in, t_out = t_in.to(self.device), t_out.to(self.device)
                        raw = student_layer(t_in)
                        s_out = raw[0] if isinstance(raw, tuple) else raw
                        loss = self.loss_fn(s_out, t_out)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        batch_losses.append(loss.item())
                else:
                    for x, _ in dataloader:
                        if self.device is not None:
                            x = x.to(self.device)
                        with torch.no_grad():
                            self.teacher(x)
                        self.student(x)
                        t_out, s_out = teacher_out["out"], student_out["out"]
                        loss = self.loss_fn(s_out, t_out)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        batch_losses.append(loss.item())

                mean_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(mean_loss)
                if getattr(student_layer, "enable_pruning", False):
                    student_layer.pruning_layer.post_epoch_function(epoch, n_epochs)
                val_loss: float | None = None
                if val_dataloader is not None:
                    val_loss = self.val_layer_loss(teacher_layer, student_layer, val_dataloader)
                if epoch_callback is not None:
                    epoch_callback(epoch, mean_loss, val_loss, student_layer_name)
        finally:
            for h in hooks:
                h.remove()
            if tmpdir is not None:
                tmpdir.cleanup()
            for name, param in self.student.named_parameters():
                if name in frozen:
                    param.requires_grad_(frozen[name])

        return epoch_losses

    def distill_all(
        self,
        dataloader: Iterable,
        optimizer_factory: Callable[[Iterable], torch.optim.Optimizer],
        n_epochs: int,
        layer_names: list[tuple[str, str]] | None = None,
        val_dataloader: Iterable | None = None,
        epoch_callback: Callable[[int, float, float | None, str], None] | None = None,
        layer_callback: Callable[[str, str, list[float]], None] | None = None,
    ) -> dict[tuple[str, str], list[float]]:
        """
        Distill all PQ layers sequentially front-to-back.

        Args:
            dataloader:        Iterable of ``(inputs, targets)`` batches.
            optimizer_factory: Callable ``(params) -> Optimizer``. Called fresh
                               for each layer so state never carries over.
            n_epochs:          Number of epochs (full dataloader passes) per layer.
            layer_names:       List of ``(teacher_layer_name, student_layer_name)``
                               tuples to distill. Defaults to pairing each
                               ``PQWeightBiasBase`` submodule name with itself.
            val_dataloader:    Optional validation dataloader passed through to
                               each ``distill_layer`` call.
            epoch_callback:    Passed through to ``distill_layer``. Called after
                               each epoch with
                               ``(epoch, train_loss, val_loss, layer_name)``
                               where ``val_loss`` is ``None`` when no
                               ``val_dataloader`` is given.
            layer_callback:    Called after each layer completes with
                               ``(teacher_layer_name, student_layer_name, losses)``.

        Returns:
            Dict mapping ``(teacher_layer_name, student_layer_name)`` to list
            of per-epoch mean losses.
        """
        if layer_names is not None:
            pairs = layer_names
        else:
            names = pq_layer_names(self.student)
            pairs = [(n, n) for n in names]
        history: dict[tuple[str, str], list[float]] = {}

        for t_name, s_name in pairs:
            losses = self.distill_layer(
                t_name,
                s_name,
                dataloader,
                optimizer_factory,
                n_epochs,
                val_dataloader=val_dataloader,
                epoch_callback=epoch_callback,
            )
            history[(t_name, s_name)] = losses
            if layer_callback is not None:
                layer_callback(t_name, s_name, losses)

        return history


class ModelDistiller:
    """
    Knowledge distillation at the model output level.

    The teacher's logits are used as soft targets for the student via KL
    divergence loss. Because the teacher (16 raw classes) and student (10
    merged classes) have different output sizes, a raw-to-merged mapping
    tensor is used to collapse the teacher logits to 10 classes before
    computing the loss.

    Args:
        teacher: Full-precision reference model (16-class output). Set to
                 eval() with gradients disabled.
        student: Student model (10-class output).
        teacher_transform: Optional callable applied to teacher logits before
                       the distillation loss is computed. Use this to map
                       teacher outputs to the student's output space, e.g. to
                       collapse classes. If ``None``, teacher logits are used
                       as-is (teacher and student must then have matching output
                       shapes).
        loss_fn:       Which distillation loss to use. One of:
                       ``"kl_ce"`` (default) — ``alpha * KL(student || teacher)
                       + (1 - alpha) * CE(student, labels)`` with temperature
                       scaling. Classic Hinton et al. formulation.
                       ``"kl"`` — KL divergence only, no hard label term.
                       Useful when ground-truth labels are unavailable or noisy.
                       ``"mse"`` — mean squared error directly on the logits,
                       no temperature scaling. Simpler baseline; equivalent to
                       KL under a uniform teacher distribution.
        temperature:   Softmax temperature for soft targets (default 4.0).
                       Only used by ``"kl_ce"`` and ``"kl"``.
        alpha:         Weight of the KL distillation loss (default 0.7).
                       The remaining ``(1 - alpha)`` weight is given to the
                       hard cross-entropy loss. Only used by ``"kl_ce"``.
        precompute_teacher_outputs: If True (default), run a single inference
                       pass of the teacher over the entire dataset before
                       distillation begins, caching transformed teacher logits
                       to disk. All epoch loops then
                       use this cached loader and never call the teacher again.
                       Set to False only when the dataloader uses stochastic
                       augmentations that must remain live during training.
        prefetch_workers: Number of DataLoader worker processes used to
                       prefetch cached batches from disk when
                       ``precompute_teacher_outputs=True``. Defaults to 2.
        cache_dir:     Directory under which the temporary cache is created
                       when ``precompute_teacher_outputs=True``. Defaults to
                       the current working directory. Avoid ``/tmp`` on Linux
                       as it is typically a RAM-backed ``tmpfs``.
    """

    LOSS_FN_OPTIONS = ("kl_ce", "kl", "mse")

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        pq_config,
        device: torch.device | None,
        teacher_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        loss_fn: str = "kl_ce",
        temperature: float = 4.0,
        alpha: float = 0.7,
        precompute_teacher_outputs: bool = True,
        prefetch_workers: int = 2,
        cache_dir: str | None = None,
    ):
        if loss_fn not in self.LOSS_FN_OPTIONS:
            raise ValueError(f"loss_fn must be one of {self.LOSS_FN_OPTIONS}, got '{loss_fn}'")

        self.teacher = teacher
        self.student = student
        self.pq_config = pq_config
        self.teacher_transform = teacher_transform
        self.loss_fn = loss_fn
        self.T = temperature
        self.alpha = alpha
        self._precompute_teacher_outputs = precompute_teacher_outputs
        self.prefetch_workers = prefetch_workers
        self.cache_dir = cache_dir
        self.device = device
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def precompute_teacher_outputs(
        self,
        dataloader: Iterable,
        cache_dir: str,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Run one teacher inference pass and cache ``(x, teacher_output, y)`` to disk."""
        n_batches = 0
        self.teacher.eval()
        with torch.no_grad():
            for x, y in dataloader:
                if self.device is not None:
                    x = x.to(self.device)
                teacher_logits = self.teacher(x)
                teacher_out = self.teacher_transform(teacher_logits) if self.teacher_transform else teacher_logits
                torch.save(
                    (x.cpu(), teacher_out.cpu(), y.cpu()),
                    Path(cache_dir) / f"{n_batches:08d}.pt",
                )
                n_batches += 1

        dataset = CachedDataset(cache_dir, n_batches)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.prefetch_workers,
            collate_fn=lambda b: b[0],
        )

    def kl_divergence(self, student_logits: torch.Tensor, teacher_merged: torch.Tensor) -> torch.Tensor:
        _, C, _, _ = student_logits.shape
        flat_s = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
        flat_t = teacher_merged.permute(0, 2, 3, 1).reshape(-1, C)
        log_p_s = F.log_softmax(flat_s / self.T, dim=1)
        p_t = F.softmax(flat_t / self.T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T**2)

    def loss_kl_ce(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """alpha * KL(student || teacher) + (1 - alpha) * CE(student, labels)."""
        kl_loss = self.kl_divergence(student_logits, teacher_logits)
        ce_loss = F.cross_entropy(student_logits, labels, ignore_index=-1)
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss

    def loss_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence only, no hard label term."""
        return self.kl_divergence(student_logits, teacher_logits)

    def loss_mse(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """MSE directly on logits, no temperature scaling."""
        return F.mse_loss(student_logits, teacher_logits)

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_fn == "kl_ce":
            return self.loss_kl_ce(student_logits, teacher_logits, labels)
        if self.loss_fn == "kl":
            return self.loss_kl(student_logits, teacher_logits)
        # mse
        return self.loss_mse(student_logits, teacher_logits)

    def run_val_epoch(
        self,
        val_dataloader: Iterable,
    ) -> float:
        """Compute mean validation loss over one pass of val_dataloader (no grad, eval mode).

        When ``precompute_teacher_outputs=True`` the dataloader yields
        ``(x, teacher_logits, y)`` and the cached teacher output is used directly.
        Otherwise it yields ``(x, y)`` and the teacher is run live.
        """
        self.student.eval()
        batch_losses: list[float] = []
        with torch.no_grad():
            for batch in val_dataloader:
                if self._precompute_teacher_outputs:
                    x, teacher_logits, y = batch
                    if self.device is not None:
                        x, teacher_logits, y = x.to(self.device), teacher_logits.to(self.device), y.to(self.device)
                else:
                    x, y = batch
                    if self.device is not None:
                        x, y = x.to(self.device), y.to(self.device)
                    teacher_logits = self.teacher(x)
                    teacher_logits = self.teacher_transform(teacher_logits) if self.teacher_transform else teacher_logits
                student_logits = self.student(x)
                loss = self.compute_loss(student_logits, teacher_logits, y)
                batch_losses.append(loss.item())
        self.student.train()
        return sum(batch_losses) / len(batch_losses)

    def run_epoch(
        self,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.student.train()
        batch_losses: list[float] = []

        for batch in dataloader:
            if self._precompute_teacher_outputs:
                x, teacher_logits, y = batch
                if self.device is not None:
                    x, teacher_logits, y = x.to(self.device), teacher_logits.to(self.device), y.to(self.device)
            else:
                x, y = batch
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                    teacher_logits = self.teacher_transform(teacher_logits) if self.teacher_transform else teacher_logits

            student_logits = self.student(x)
            loss = self.compute_loss(student_logits, teacher_logits, y)
            loss = get_model_losses(self.student, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        return sum(batch_losses) / len(batch_losses)

    def distill(
        self,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        val_dataloader: Iterable | None = None,
        epoch_callback: Callable[[int, float, float | None], None] | None = None,
    ) -> list[float]:
        """
        Run model distillation following the PQuantML training pipeline.

        Args:
            dataloader:       Dataloader provided by user.
            optimizer:        Optimizer for student parameters.
            val_dataloader:   Optional validation dataloader. When provided, a
                              no-grad eval pass is run after every training epoch
                              and the resulting mean loss is passed to
                              ``epoch_callback``.
            epoch_callback:   Called after each epoch with
                              ``(epoch, train_loss, val_loss)`` where
                              ``val_loss`` is ``None`` when no ``val_dataloader``
                              is given.

        Returns:
            List of per-epoch mean training losses.
        """
        training_parameters = self.pq_config.training_parameters

        tmpdir = None
        tmpdir_val = None
        epoch_losses: list[float] = []
        global_epoch = 0

        try:
            if self._precompute_teacher_outputs:
                tmpdir = tempfile.TemporaryDirectory(prefix="mdistil_", dir=self.cache_dir or str(Path.cwd()))
                dataloader = self.precompute_teacher_outputs(dataloader, tmpdir.name)
                if val_dataloader is not None:
                    tmpdir_val = tempfile.TemporaryDirectory(prefix="mdistil_val_", dir=self.cache_dir or str(Path.cwd()))
                    val_dataloader = self.precompute_teacher_outputs(val_dataloader, tmpdir_val.name, shuffle=False)

            for e in range(training_parameters.pretraining_epochs):
                pre_epoch_functions(self.student, e, training_parameters.pretraining_epochs)
                mean_loss = self.run_epoch(dataloader, optimizer)
                epoch_losses.append(mean_loss)
                post_epoch_functions(self.student, e, training_parameters.pretraining_epochs)
                val_loss: float | None = None
                if val_dataloader is not None:
                    val_loss = self.run_val_epoch(val_dataloader)
                if epoch_callback is not None:
                    epoch_callback(global_epoch, mean_loss, val_loss)
                global_epoch += 1

            post_pretrain_functions(self.student, self.pq_config, train_loader=dataloader)

            for _ in range(training_parameters.rounds):
                for e in range(training_parameters.epochs):
                    pre_epoch_functions(self.student, e, training_parameters.epochs)
                    mean_loss = self.run_epoch(dataloader, optimizer)
                    epoch_losses.append(mean_loss)
                    post_epoch_functions(self.student, e, training_parameters.epochs)
                    val_loss = None
                    if val_dataloader is not None:
                        val_loss = self.run_val_epoch(val_dataloader)
                    if epoch_callback is not None:
                        epoch_callback(global_epoch, mean_loss, val_loss)
                    global_epoch += 1
                post_round_functions(self.student)

            pre_finetune_functions(self.student)
            for e in range(training_parameters.fine_tuning_epochs):
                pre_epoch_functions(self.student, e, training_parameters.fine_tuning_epochs)
                mean_loss = self.run_epoch(dataloader, optimizer)
                epoch_losses.append(mean_loss)
                val_loss = None
                if val_dataloader is not None:
                    val_loss = self.run_val_epoch(val_dataloader)
                if epoch_callback is not None:
                    epoch_callback(global_epoch, mean_loss, val_loss)
                post_epoch_functions(self.student, e, training_parameters.fine_tuning_epochs)
                global_epoch += 1
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()
            if tmpdir_val is not None:
                tmpdir_val.cleanup()

        return epoch_losses
