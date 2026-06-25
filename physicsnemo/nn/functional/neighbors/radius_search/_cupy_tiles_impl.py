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

"""Experimental CuPy/CUDA tiled radius-search implementation.

This backend is intentionally separate from the production torch and warp
implementations. It is a sample kernel for testing a query-local tiled schedule:
support points are sorted into compact cell ranges, queries are sorted by cell,
and persistent CUDA blocks pull spatial query tiles while assigning one warp to
each query.
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


_CELL_HASH_PRIMES = (73856093, 19349663, 83492791)
_BATCH_HASH_PRIME = 2654435761
_CELL_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)


_RADIUS_SEARCH_KERNEL = r"""
extern "C" __global__
void radius_search_tiles(
    const float* __restrict__ points,
    const float* __restrict__ queries,
    const long long* __restrict__ sorted_point_ids,
    const long long* __restrict__ unique_cell_keys,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const long long* __restrict__ sorted_query_ids,
    const long long* __restrict__ neighbor_cell_keys,
    int* __restrict__ tile_counter,
    int* __restrict__ indices,
    float* __restrict__ distances,
    int* __restrict__ counts,
    const int num_tiles,
    const int queries_per_tile,
    const int num_queries,
    const int num_points,
    const int max_points,
    const int num_unique_cells,
    const float radius_sq,
    const int write_dists
) {
    constexpr int warp_size = 32;
    constexpr int warps_per_block = 8;

    __shared__ int tile;
    __shared__ int warp_counts[warps_per_block];

    const int lane = threadIdx.x & (warp_size - 1);
    const int warp_id = threadIdx.x / warp_size;

    while (true) {
        if (threadIdx.x == 0) {
            tile = atomicAdd(tile_counter, 1);
        }
        __syncthreads();

        if (tile >= num_tiles) {
            break;
        }

        const int tile_begin = tile * queries_per_tile;
        const int tile_end_bound = tile_begin + queries_per_tile;
        const int tile_end =
            tile_end_bound < num_queries ? tile_end_bound : num_queries;

        for (int tile_offset = warp_id; tile_begin + tile_offset < tile_end;
             tile_offset += warps_per_block) {
            const long long query_id_ll = sorted_query_ids[tile_begin + tile_offset];
            const int query_id = static_cast<int>(query_id_ll);
            const float qx = queries[query_id * 3 + 0];
            const float qy = queries[query_id * 3 + 1];
            const float qz = queries[query_id * 3 + 2];

            if (lane == 0) {
                warp_counts[warp_id] = 0;
            }
            __syncwarp();

            #pragma unroll
            for (int neighbor = 0; neighbor < 27; ++neighbor) {
                const long long key = neighbor_cell_keys[query_id * 27 + neighbor];

                int lo = 0;
                int hi = num_unique_cells;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (unique_cell_keys[mid] < key) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                if (lo >= num_unique_cells || unique_cell_keys[lo] != key) {
                    continue;
                }

                const int start = cell_start[lo];
                const int end = cell_end[lo];
                for (int sorted_point_offset = start + lane;
                     sorted_point_offset < end;
                     sorted_point_offset += warp_size) {
                    const long long point_id_ll = sorted_point_ids[sorted_point_offset];
                    const int point_id = static_cast<int>(point_id_ll);
                    const float dx = points[point_id * 3 + 0] - qx;
                    const float dy = points[point_id * 3 + 1] - qy;
                    const float dz = points[point_id * 3 + 2] - qz;
                    const float dist_sq = dx * dx + dy * dy + dz * dz;

                    if (dist_sq <= radius_sq) {
                        const int slot = atomicAdd(&warp_counts[warp_id], 1);
                        if (slot < max_points) {
                            const int out_offset = query_id * max_points + slot;
                            indices[out_offset] = point_id % num_points;
                            if (write_dists) {
                                distances[out_offset] = sqrtf(dist_sq);
                            }
                        }
                    }
                }
            }

            __syncwarp();
            if (lane == 0) {
                const int found = warp_counts[warp_id];
                counts[query_id] = found < max_points ? found : max_points;
            }
            __syncwarp();
        }

        __syncthreads();
    }
}
"""


@lru_cache(maxsize=1)
def _get_radius_search_kernel():
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search cupy_tiles implementation requires cupy>=13.6.0"
        )
    module = cp.RawModule(
        code=_RADIUS_SEARCH_KERNEL,
        options=("--std=c++17",),
        name_expressions=("radius_search_tiles",),
    )
    return module.get_function("radius_search_tiles")


def _require_cupy() -> None:
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search cupy_tiles implementation requires cupy>=13.6.0"
        )


def _cell_keys(
    coords: torch.Tensor,
    batch_ids: torch.Tensor,
) -> torch.Tensor:
    """Hash integer cell coordinates and batch ids into signed int64 keys."""
    coords = coords.to(torch.int64)
    batch_ids = batch_ids.to(torch.int64)
    return (
        coords[:, 0] * _CELL_HASH_PRIMES[0]
        ^ coords[:, 1] * _CELL_HASH_PRIMES[1]
        ^ coords[:, 2] * _CELL_HASH_PRIMES[2]
        ^ batch_ids * _BATCH_HASH_PRIME
    )


def _as_cupy(tensor: torch.Tensor):
    """Create a zero-copy CuPy view of a CUDA torch tensor."""
    assert cp is not None
    return cp.from_dlpack(tensor.contiguous())


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
        if was_unbatched:
            return output.squeeze(0)
        return output

    batch = torch.arange(batch_size, device=points.device)[:, None, None]
    gathered = points[batch, indices.to(torch.long)]
    valid = (
        torch.arange(max_points, device=points.device)[None, None, :]
        < counts[:, :, None]
    )
    gathered = torch.where(valid[..., None], gathered, torch.zeros_like(gathered))
    if was_unbatched:
        return gathered.squeeze(0)
    return gathered.reshape(batch_size, num_queries, max_points, point_dim)


def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Experimental persistent spatial-tile CUDA radius search.

    This implementation supports only the static ``max_points`` path. It uses
    CuPy RawKernel for the search kernel and torch operations for sorting and
    compact cell-range construction.
    """

    _require_cupy()

    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.device.type != "cuda":
        raise ValueError("cupy_tiles radius_search implementation requires CUDA input")
    if max_points is None:
        raise ValueError("cupy_tiles radius_search implementation requires max_points")
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    input_points = points
    input_queries = queries
    points, queries, was_unbatched = validate_inputs(points, queries)
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError("cupy_tiles radius_search implementation requires 3D points")

    if points.dtype != torch.float32:
        points_kernel = points.to(torch.float32)
    else:
        points_kernel = points
    if queries.dtype != torch.float32:
        queries_kernel = queries.to(torch.float32)
    else:
        queries_kernel = queries

    points_kernel = points_kernel.contiguous()
    queries_kernel = queries_kernel.contiguous()

    batch_size, num_points, _ = points_kernel.shape
    _, num_queries, _ = queries_kernel.shape
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
    if return_dists:
        distances = torch.zeros(
            (batch_size, num_queries, max_points),
            dtype=torch.float32,
            device=points.device,
        )
    else:
        distances = torch.empty(0, dtype=torch.float32, device=points.device)

    if total_queries > 0 and total_points > 0:
        flat_points = points_kernel.reshape(total_points, 3)
        flat_queries = queries_kernel.reshape(total_queries, 3)

        point_batch = torch.arange(
            batch_size, device=points.device
        ).repeat_interleave(num_points)
        query_batch = torch.arange(
            batch_size, device=points.device
        ).repeat_interleave(num_queries)

        point_cells = torch.floor(flat_points / radius).to(torch.int64)
        query_cells = torch.floor(flat_queries / radius).to(torch.int64)

        point_keys = _cell_keys(point_cells, point_batch)
        sorted_point_keys, sorted_point_ids = torch.sort(point_keys)
        unique_cell_keys, cell_counts = torch.unique_consecutive(
            sorted_point_keys, return_counts=True
        )
        cell_end = torch.cumsum(cell_counts, dim=0).to(torch.int32)
        cell_start = torch.empty_like(cell_end)
        cell_start[0] = 0
        cell_start[1:] = cell_end[:-1]

        query_keys = _cell_keys(query_cells, query_batch)
        _, sorted_query_ids = torch.sort(query_keys)

        offsets = torch.tensor(_CELL_OFFSETS, dtype=torch.int64, device=points.device)
        neighbor_cells = query_cells[:, None, :] + offsets[None, :, :]
        neighbor_batch = query_batch[:, None].expand(-1, len(_CELL_OFFSETS))
        neighbor_cell_keys = _cell_keys(
            neighbor_cells.reshape(-1, 3), neighbor_batch.reshape(-1)
        ).reshape(total_queries, len(_CELL_OFFSETS))

        queries_per_tile = 128
        num_tiles = (total_queries + queries_per_tile - 1) // queries_per_tile
        tile_counter = torch.zeros(1, dtype=torch.int32, device=points.device)

        device_index = points.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        kernel = _get_radius_search_kernel()
        current_stream = torch.cuda.current_stream(device_index).cuda_stream
        cupy_stream = cp.cuda.ExternalStream(current_stream)

        with cp.cuda.Device(device_index), cupy_stream:
            device_props = cp.cuda.runtime.getDeviceProperties(device_index)
            sm_count = int(device_props["multiProcessorCount"])
            blocks = max(1, min(num_tiles, sm_count * 4))
            threads = 256
            kernel(
                (blocks,),
                (threads,),
                (
                    _as_cupy(flat_points),
                    _as_cupy(flat_queries),
                    _as_cupy(sorted_point_ids.to(torch.int64)),
                    _as_cupy(unique_cell_keys.to(torch.int64)),
                    _as_cupy(cell_start),
                    _as_cupy(cell_end),
                    _as_cupy(sorted_query_ids.to(torch.int64)),
                    _as_cupy(neighbor_cell_keys.to(torch.int64)),
                    _as_cupy(tile_counter),
                    _as_cupy(indices.reshape(total_queries, max_points)),
                    _as_cupy(distances.reshape(total_queries, max_points))
                    if return_dists
                    else _as_cupy(
                        torch.empty(0, dtype=torch.float32, device=points.device)
                    ),
                    _as_cupy(counts.reshape(total_queries)),
                    num_tiles,
                    queries_per_tile,
                    total_queries,
                    num_points,
                    max_points,
                    int(unique_cell_keys.numel()),
                    float(radius * radius),
                    int(return_dists),
                ),
            )

    if return_points:
        original_points, _, _ = validate_inputs(input_points, input_queries)
        points_out = _gather_points(original_points, indices, counts, was_unbatched)
    else:
        points_out = torch.empty(
            0, max_points, 3, device=points.device, dtype=points.dtype
        )

    if was_unbatched:
        indices = indices.squeeze(0)
        if return_dists:
            distances = distances.squeeze(0)

    return indices, points_out, distances


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """CuPy tiled-backend entry point for radius search with formatted returns."""
    indices, points_out, distances = radius_search_impl(
        points, queries, radius, max_points, return_dists, return_points
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
