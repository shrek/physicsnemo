# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhysicsNeMo adapter for the packaged physicsnemo-ops radius search."""

from __future__ import annotations

import torch

from .utils import format_returns, validate_inputs

try:
    from physicsnemo_ops.torch import radius_search as _ops_radius_search
except ImportError as exc:
    _ops_radius_search = None
    _OPS_IMPORT_ERROR: ImportError | None = exc
else:
    _OPS_IMPORT_ERROR = None


def _gather_points(
    points: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected points and return the valid-prefix mask."""
    batch_size, num_points, point_dim = points.shape
    _, num_queries, max_points = indices.shape
    valid = (
        torch.arange(max_points, device=points.device)[None, None, :]
        < counts[..., None]
    )
    if num_points == 0:
        selected = torch.zeros(
            batch_size,
            num_queries,
            max_points,
            point_dim,
            dtype=points.dtype,
            device=points.device,
        )
        return selected, valid

    batch = torch.arange(batch_size, device=points.device)[:, None, None]
    selected = points[batch, indices.to(torch.long)]
    selected = torch.where(valid[..., None], selected, torch.zeros_like(selected))
    return selected, valid


def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run packaged radius search and compose PhysicsNeMo's rich returns."""
    if _ops_radius_search is None:
        raise ImportError(
            "physicsnemo radius_search implementation 'physicsnemo_ops' requires "
            "the physicsnemo-ops package"
        ) from _OPS_IMPORT_ERROR
    if max_points is None:
        raise ValueError(
            "physicsnemo_ops radius_search implementation requires max_points"
        )
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")

    same_input = points is queries
    if not torch.compiler.is_compiling():
        same_input = same_input or (
            points.shape == queries.shape
            and points.stride() == queries.stride()
            and points.data_ptr() == queries.data_ptr()
        )

    points, queries, was_unbatched = validate_inputs(points, queries)
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError(
            "physicsnemo_ops radius_search implementation requires 3D points"
        )

    points_kernel = points.to(torch.float32).contiguous()
    if same_input:
        queries_kernel = points_kernel
    else:
        queries_kernel = queries.to(torch.float32).contiguous()

    indices, counts = _ops_radius_search(
        points_kernel,
        queries_kernel,
        float(radius),
        max_points,
    )

    need_selected = return_points or return_dists
    if need_selected:
        selected_points, valid = _gather_points(points, indices, counts)
    else:
        selected_points = torch.empty(
            0, max_points, 3, dtype=points.dtype, device=points.device
        )
        valid = torch.empty(0, dtype=torch.bool, device=points.device)

    if return_dists:
        distances = torch.linalg.vector_norm(
            selected_points.float() - queries.float().unsqueeze(-2), dim=-1
        )
        distances = torch.where(valid, distances, torch.zeros_like(distances))
    else:
        distances = torch.empty(0, dtype=torch.float32, device=points.device)

    if was_unbatched:
        indices = indices.squeeze(0)
        if need_selected:
            selected_points = selected_points.squeeze(0)
        if return_dists:
            distances = distances.squeeze(0)

    return indices, selected_points, distances


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """Run the physicsnemo-ops backend with PhysicsNeMo-formatted returns."""
    indices, points_out, distances = radius_search_impl(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
