// SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
