# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape-batched Muon optimizer.

This preserves PyTorch Muon's per-parameter state and update semantics, but
groups repeated 2-D parameter shapes so the Newton-Schulz orthogonalization uses
batched matrix multiplies instead of many small per-parameter GEMMs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5
EPS = 1e-7


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Match ``torch.optim.Muon`` learning-rate adjustment."""
    a_dim, b_dim = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, a_dim / b_dim))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(a_dim, b_dim))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class ShapeBatchedMuon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz work batched by matrix shape.

    The optimizer state intentionally matches PyTorch Muon's state layout:
    every parameter stores a ``momentum_buffer`` tensor. This lets ordinary
    ``Optimizer.load_state_dict`` handle stock-Muon and ShapeBatchedMuon state
    migration when the parameter ordering is unchanged.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (
            DEFAULT_A,
            DEFAULT_B,
            DEFAULT_C,
        ),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if len(ns_coefficients) != 3:
            raise ValueError("Coefficients must be a tuple of exactly 3 values")
        if ns_steps >= 100:
            raise ValueError(
                "Number of steps must be less than 100 for computational efficiency"
            )
        if adjust_lr_fn is not None and adjust_lr_fn not in {
            "original",
            "match_rms_adamw",
        }:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "ShapeBatchedMuon only supports 2-D parameters whereas "
                        f"we found a parameter with size: {param.size()}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._step_group(group)

        return loss

    def _step_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        if isinstance(lr, Tensor):
            lr = float(lr.item())
        else:
            lr = float(lr)
        weight_decay = float(group["weight_decay"])
        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        a_coeff, b_coeff, c_coeff = group["ns_coefficients"]
        eps = float(group["eps"])
        ns_steps = int(group["ns_steps"])
        adjust_lr_fn = group["adjust_lr_fn"]

        grouped_updates: dict[
            tuple[torch.device, torch.dtype, tuple[int, int]],
            list[tuple[Tensor, Tensor, bool]],
        ] = defaultdict(list)

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.ndim != 2:
                raise ValueError("Param gradient must be a 2-D matrix")
            if torch.is_complex(param):
                raise RuntimeError("Muon does not support complex parameters")
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    grad, memory_format=torch.preserve_format
                )

            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - momentum)
            update = grad.lerp(buf, momentum) if nesterov else buf

            transposed = update.size(0) > update.size(1)
            oriented = update.T if transposed else update
            key = (oriented.device, oriented.dtype, tuple(oriented.shape))
            grouped_updates[key].append((param, oriented, transposed))

        for items in grouped_updates.values():
            x = torch.stack([item[1].to(torch.bfloat16) for item in items], dim=0)
            norms = x.flatten(1).norm(dim=1).clamp(min=eps).view(-1, 1, 1)
            x.div_(norms)

            for _ in range(ns_steps):
                gram = x @ x.transpose(-2, -1)
                gram_update = torch.baddbmm(
                    gram, gram, gram, beta=b_coeff, alpha=c_coeff
                )
                x = torch.baddbmm(x, gram_update, x, beta=a_coeff)

            for i, (param, _oriented, transposed) in enumerate(items):
                update = x[i].T if transposed else x[i]
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update, alpha=-adjusted_lr)
