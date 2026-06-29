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

import torch
from physicsnemo.core.version_check import check_version_spec

from .utils import format_returns, validate_inputs

CUPY_AVAILABLE = check_version_spec("cupy", "13.6.0", hard_fail=False)

if CUPY_AVAILABLE:
    cp = importlib.import_module("cupy")
else:
    cp = None


_MORTON_CELL_POINTS_KERNEL = r"""
namespace {

constexpr int COORD_BITS = 20;
constexpr int BATCH_BITS = 3;
constexpr long long COORD_BIAS = 1LL << (COORD_BITS - 1);
constexpr unsigned int COORD_MASK = (1U << COORD_BITS) - 1U;
constexpr unsigned long long BATCH_LIMIT = 1ULL << BATCH_BITS;
constexpr int BATCH_SHIFT = 3 * COORD_BITS;

__device__ __forceinline__ unsigned long long split_by_3_20(unsigned int x) {
    unsigned long long v = static_cast<unsigned long long>(x & COORD_MASK);
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
    v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2)) & 0x1249249249249249ULL;
    return v;
}

__device__ __forceinline__ long long morton_cell_key(
    const long long cx,
    const long long cy,
    const long long cz,
    const long long batch
) {
    const unsigned int ux = static_cast<unsigned int>(cx + COORD_BIAS) & COORD_MASK;
    const unsigned int uy = static_cast<unsigned int>(cy + COORD_BIAS) & COORD_MASK;
    const unsigned int uz = static_cast<unsigned int>(cz + COORD_BIAS) & COORD_MASK;
    const unsigned long long morton =
        split_by_3_20(ux)
        | (split_by_3_20(uy) << 1)
        | (split_by_3_20(uz) << 2);
    return static_cast<long long>(
        (static_cast<unsigned long long>(batch) << BATCH_SHIFT) | morton);
}

__device__ __forceinline__ int lower_bound_key(
    const long long* __restrict__ keys,
    const int num_keys,
    const long long key
) {
    int lo = 0;
    int hi = num_keys;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (keys[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

}  // namespace

extern "C" __global__
void compute_point_morton_keys(
    const float* __restrict__ points,
    long long* __restrict__ point_keys,
    int* __restrict__ point_ids,
    int* __restrict__ overflow,
    const int total_points,
    const int num_points,
    const float radius
) {
    const int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= total_points) {
        return;
    }

    const long long batch = point_id / num_points;
    const float px = points[point_id * 3 + 0];
    const float py = points[point_id * 3 + 1];
    const float pz = points[point_id * 3 + 2];
    const double inv_radius = 1.0 / static_cast<double>(radius);
    const long long cx = static_cast<long long>(
        floor(static_cast<double>(px) * inv_radius));
    const long long cy = static_cast<long long>(
        floor(static_cast<double>(py) * inv_radius));
    const long long cz = static_cast<long long>(
        floor(static_cast<double>(pz) * inv_radius));

    const long long min_coord = -COORD_BIAS;
    const long long max_coord = COORD_BIAS - 1;
    if (batch < 0 || static_cast<unsigned long long>(batch) >= BATCH_LIMIT
        || cx < min_coord || cx > max_coord
        || cy < min_coord || cy > max_coord
        || cz < min_coord || cz > max_coord) {
        atomicExch(overflow, 1);
    }

    point_keys[point_id] = morton_cell_key(cx, cy, cz, batch);
    point_ids[point_id] = point_id;
}

extern "C" __global__
void radius_search_morton_cell_points(
    const float* __restrict__ points,
    const float* __restrict__ queries,
    const long long* __restrict__ unique_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_counts,
    const int* __restrict__ sorted_point_ids,
    int* __restrict__ indices,
    float* __restrict__ output_points,
    float* __restrict__ distances,
    int* __restrict__ counts,
    int* __restrict__ overflow,
    const int total_queries,
    const int queries_per_batch,
    const int num_points,
    const int max_points,
    const int num_cells,
    const float radius,
    const float radius_sq,
    const int write_points,
    const int write_dists
) {
    constexpr int warp_size = 32;
    constexpr int warps_per_block = 8;
    constexpr unsigned int full_mask = 0xffffffffU;

    const int lane = threadIdx.x & (warp_size - 1);
    const int warp_id = threadIdx.x / warp_size;
    const int query_id = blockIdx.x * warps_per_block + warp_id;
    if (query_id >= total_queries) {
        return;
    }

    const float qx = queries[query_id * 3 + 0];
    const float qy = queries[query_id * 3 + 1];
    const float qz = queries[query_id * 3 + 2];
    const long long query_batch = query_id / queries_per_batch;
    const double inv_radius = 1.0 / static_cast<double>(radius);
    const long long qcx = static_cast<long long>(
        floor(static_cast<double>(qx) * inv_radius));
    const long long qcy = static_cast<long long>(
        floor(static_cast<double>(qy) * inv_radius));
    const long long qcz = static_cast<long long>(
        floor(static_cast<double>(qz) * inv_radius));

    const long long min_coord = -COORD_BIAS;
    const long long max_coord = COORD_BIAS - 1;
    if (lane == 0) {
        if (query_batch < 0
            || static_cast<unsigned long long>(query_batch) >= BATCH_LIMIT
            || qcx - 1 < min_coord || qcx + 1 > max_coord
            || qcy - 1 < min_coord || qcy + 1 > max_coord
            || qcz - 1 < min_coord || qcz + 1 > max_coord) {
            atomicExch(overflow, 1);
        }
    }

    int found = 0;
    const int out_base = query_id * max_points;

    for (int dx_cell = -1; dx_cell <= 1 && found < max_points; ++dx_cell) {
        for (int dy_cell = -1; dy_cell <= 1 && found < max_points; ++dy_cell) {
            for (int dz_cell = -1; dz_cell <= 1 && found < max_points; ++dz_cell) {
                int cell_slot = -1;
                if (lane == 0) {
                    const long long key = morton_cell_key(
                        qcx + dx_cell,
                        qcy + dy_cell,
                        qcz + dz_cell,
                        query_batch);
                    const int candidate_slot = lower_bound_key(
                        unique_keys,
                        num_cells,
                        key);
                    if (candidate_slot < num_cells && unique_keys[candidate_slot] == key) {
                        cell_slot = candidate_slot;
                    }
                }
                cell_slot = __shfl_sync(full_mask, cell_slot, 0);
                if (cell_slot < 0) {
                    continue;
                }

                const int cell_start = cell_offsets[cell_slot];
                const int cell_count = cell_counts[cell_slot];
                const int cell_end = cell_start + cell_count;
                for (int candidate_base = cell_start;
                     candidate_base < cell_end && found < max_points;
                     candidate_base += warp_size) {
                    const int candidate_offset = candidate_base + lane;
                    int point_id = -1;
                    float dist_sq = 0.0f;
                    bool hit = false;
                    if (candidate_offset < cell_end) {
                        point_id = sorted_point_ids[candidate_offset];
                        const float px = points[point_id * 3 + 0];
                        const float py = points[point_id * 3 + 1];
                        const float pz = points[point_id * 3 + 2];
                        const float xdiff = px - qx;
                        const float ydiff = py - qy;
                        const float zdiff = pz - qz;
                        dist_sq = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;
                        hit = dist_sq <= radius_sq;
                    }

                    const unsigned int hit_mask = __ballot_sync(full_mask, hit);
                    const int num_hits = __popc(hit_mask);
                    if (num_hits == 0) {
                        continue;
                    }

                    const int space = max_points - found;
                    const int write_hits = num_hits < space ? num_hits : space;
                    const unsigned int lower_lane_mask = (1U << lane) - 1U;
                    const int hit_rank = __popc(hit_mask & lower_lane_mask);
                    if (hit && hit_rank < write_hits) {
                        const int out_offset = out_base + found + hit_rank;
                        const int local_point = point_id % num_points;
                        indices[out_offset] = local_point;
                        if (write_points) {
                            output_points[out_offset * 3 + 0] = points[point_id * 3 + 0];
                            output_points[out_offset * 3 + 1] = points[point_id * 3 + 1];
                            output_points[out_offset * 3 + 2] = points[point_id * 3 + 2];
                        }
                        if (write_dists) {
                            distances[out_offset] = sqrtf(dist_sq);
                        }
                    }

                    found += write_hits;
                }
            }
        }
    }

    if (lane == 0) {
        counts[query_id] = found;
    }
}
"""


@lru_cache(maxsize=1)
def _get_morton_cell_points_kernels():
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search morton_cell_points implementation requires "
            "cupy>=13.6.0"
        )
    module = cp.RawModule(
        code=_MORTON_CELL_POINTS_KERNEL,
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
    assert cp is not None
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
