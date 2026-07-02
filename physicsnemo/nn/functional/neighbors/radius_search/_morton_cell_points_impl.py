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

"""Experimental Morton-sorted cell-points CUDA radius-search backend.

This prototype targets the same static GeoTransolver path as
``compact_cell_points``. It removes the explicit occupied-cell hash table by:

    1) mapping points to radius-sized cells
    2) packing ``(batch, cell_x, cell_y, cell_z)`` into a sortable Morton key
    3) sorting points by key and building a compact unique-cell directory
    4) binary-searching the directory for each neighboring query cell

The packed key reserves 20 bits per coordinate and 3 bits for batch id. That is
enough for the current captured training data, but this remains an experimental
backend rather than a general replacement for the hash-table implementation.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from importlib.resources import files

import torch

from physicsnemo.core.version_check import check_version_spec

from .utils import format_returns, validate_inputs

CUPY_AVAILABLE = check_version_spec("cupy", "13.6.0", hard_fail=False)

if CUPY_AVAILABLE:
    cp = importlib.import_module("cupy")
else:
    cp = None


@lru_cache(maxsize=1)
def _load_morton_cell_points_kernel_source() -> str:
    """Load the package-shipped CUDA source for the Morton-cell backend."""
    return (
        files(__package__)
        .joinpath("kernels")
        .joinpath("morton_cell_points.cu")
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def _get_morton_cell_points_kernels():
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search morton_cell_points implementation requires "
            "cupy>=13.6.0"
        )
    module = cp.RawModule(
        code=_load_morton_cell_points_kernel_source(),
        options=("--std=c++17",),
        name_expressions=(
            "compute_point_morton_keys",
            "radius_search_morton_cell_points",
        ),
    )
    return (
        module.get_function("compute_point_morton_keys"),
        module.get_function("radius_search_morton_cell_points"),
    )


def _require_cupy() -> None:
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search morton_cell_points implementation requires "
            "cupy>=13.6.0"
        )


def _as_cupy(tensor: torch.Tensor):
    """Create a zero-copy CuPy view of a CUDA torch tensor."""
    if cp is None:
        raise RuntimeError("CuPy >= 13.6.0 is required for this backend")
    return cp.from_dlpack(tensor.detach().contiguous())


def _gather_points(
    points: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    was_unbatched: bool,
) -> torch.Tensor:
    batch_size, _, point_dim = points.shape
    _, num_queries, max_points = indices.shape
    if points.shape[1] == 0:
        output = torch.zeros(
            batch_size,
            num_queries,
            max_points,
            point_dim,
            device=points.device,
            dtype=points.dtype,
        )
        return output.squeeze(0) if was_unbatched else output

    batch = torch.arange(batch_size, device=points.device)[:, None, None]
    gathered = points[batch, indices.to(torch.long)]
    valid = (
        torch.arange(max_points, device=points.device)[None, None, :]
        < counts[:, :, None]
    )
    gathered = torch.where(valid[..., None], gathered, torch.zeros_like(gathered))
    return gathered.squeeze(0) if was_unbatched else gathered


def _build_cell_directory(
    point_keys: torch.Tensor,
    point_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sort_order = torch.argsort(point_keys)
    sorted_keys = point_keys[sort_order].contiguous()
    sorted_point_ids = point_ids[sort_order].contiguous()
    unique_keys, cell_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    cell_counts = cell_counts.to(torch.int32)
    cell_offsets = torch.empty_like(cell_counts)
    cell_offsets[0] = 0
    if cell_counts.numel() > 1:
        cell_offsets[1:] = torch.cumsum(cell_counts[:-1], dim=0, dtype=torch.int32)
    return (
        unique_keys.contiguous(),
        cell_offsets.contiguous(),
        cell_counts.contiguous(),
        sorted_point_ids,
    )


def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static first-found radius search using Morton-sorted GPU cell bins."""

    _require_cupy()

    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.device.type != "cuda":
        raise ValueError(
            "morton_cell_points radius_search implementation requires CUDA input"
        )
    if max_points is None:
        raise ValueError(
            "morton_cell_points radius_search implementation requires max_points"
        )
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    input_points = points
    input_queries = queries
    points, queries, was_unbatched = validate_inputs(points, queries)
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError(
            "morton_cell_points radius_search implementation requires 3D points"
        )

    points_kernel = (
        points.to(torch.float32) if points.dtype != torch.float32 else points
    )
    queries_kernel = (
        queries.to(torch.float32) if queries.dtype != torch.float32 else queries
    )
    points_kernel = points_kernel.contiguous()
    queries_kernel = queries_kernel.contiguous()

    batch_size, num_points, _ = points_kernel.shape
    _, num_queries, _ = queries_kernel.shape
    if batch_size >= 8:
        raise ValueError(
            "morton_cell_points prototype supports batch sizes up to 7 "
            f"with the current packed key, got {batch_size}"
        )

    total_points = batch_size * num_points
    total_queries = batch_size * num_queries

    indices = torch.zeros(
        (batch_size, num_queries, max_points),
        dtype=torch.int32,
        device=points.device,
    )
    counts = torch.zeros(
        (batch_size, num_queries), dtype=torch.int32, device=points.device
    )
    distance_workspace = torch.zeros(
        (batch_size, num_queries, max_points),
        dtype=torch.float32,
        device=points.device,
    )
    distances = (
        distance_workspace
        if return_dists
        else torch.empty(0, dtype=torch.float32, device=points.device)
    )

    use_gather_for_points = return_points and input_points.requires_grad
    write_points_in_kernel = return_points and not use_gather_for_points
    if write_points_in_kernel:
        points_out = torch.zeros(
            (batch_size, num_queries, max_points, 3),
            dtype=torch.float32,
            device=points.device,
        )
    else:
        points_out = torch.empty(
            0,
            max_points,
            3,
            dtype=torch.float32,
            device=points.device,
        )

    overflow = torch.zeros(1, dtype=torch.int32, device=points.device)
    if total_queries > 0 and total_points > 0:
        flat_points = points_kernel.reshape(total_points, 3)
        flat_queries = queries_kernel.reshape(total_queries, 3)
        point_keys = torch.empty(total_points, dtype=torch.int64, device=points.device)
        point_ids = torch.empty(total_points, dtype=torch.int32, device=points.device)

        device_index = points.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        key_kernel, search_kernel = _get_morton_cell_points_kernels()
        current_stream = torch.cuda.current_stream(device_index).cuda_stream
        cupy_stream = cp.cuda.ExternalStream(current_stream)

        with cp.cuda.Device(device_index), cupy_stream:
            threads = 256
            point_blocks = (total_points + threads - 1) // threads
            key_kernel(
                (point_blocks,),
                (threads,),
                (
                    _as_cupy(flat_points),
                    _as_cupy(point_keys),
                    _as_cupy(point_ids),
                    _as_cupy(overflow),
                    cp.int32(total_points),
                    cp.int32(num_points),
                    cp.float32(radius),
                ),
            )

        unique_keys, cell_offsets, cell_counts, sorted_point_ids = (
            _build_cell_directory(point_keys, point_ids)
        )

        with cp.cuda.Device(device_index), cupy_stream:
            warps_per_block = 8
            query_blocks = (total_queries + warps_per_block - 1) // warps_per_block
            search_kernel(
                (query_blocks,),
                (threads,),
                (
                    _as_cupy(flat_points),
                    _as_cupy(flat_queries),
                    _as_cupy(unique_keys),
                    _as_cupy(cell_offsets),
                    _as_cupy(cell_counts),
                    _as_cupy(sorted_point_ids),
                    _as_cupy(indices.reshape(total_queries, max_points)),
                    _as_cupy(points_out.reshape(-1, 3)),
                    _as_cupy(distance_workspace.reshape(total_queries, max_points)),
                    _as_cupy(counts.reshape(total_queries)),
                    _as_cupy(overflow),
                    cp.int32(total_queries),
                    cp.int32(num_queries),
                    cp.int32(num_points),
                    cp.int32(max_points),
                    cp.int32(unique_keys.numel()),
                    cp.float32(radius),
                    cp.float32(radius * radius),
                    cp.int32(write_points_in_kernel),
                    cp.int32(return_dists),
                ),
            )

        if bool(overflow.cpu().item()):
            raise ValueError(
                "morton_cell_points packed cell key overflowed; coordinates must map "
                "to cells in [-524288, 524287] and batch size must be <= 7"
            )

    if return_points:
        if use_gather_for_points:
            original_points, _, _ = validate_inputs(input_points, input_queries)
            points_out = _gather_points(original_points, indices, counts, was_unbatched)
        elif points_out.dtype != points.dtype:
            points_out = points_out.to(points.dtype)
    else:
        points_out = torch.empty(
            0, max_points, 3, device=points.device, dtype=points.dtype
        )

    if was_unbatched:
        indices = indices.squeeze(0)
        if return_points and points_out.ndim == 4:
            points_out = points_out.squeeze(0)
        if return_dists:
            distances = distances.squeeze(0)

    return indices, points_out, distances


@torch.compiler.disable
def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """morton_cell_points backend entry point for radius search with formatted returns."""
    indices, points_out, distances = radius_search_impl(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
