#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Minimal Warp object-lifetime reproducers.

These tests match the production lifetime shape:

  1. create a wp.Mesh or wp.HashGrid
  2. enqueue delay work on the same stream
  3. launch a kernel using mesh.id / grid.id while the object is alive
  4. delete the Python owner immediately after launch
  5. optionally allocate replacement objects on another stream
  6. synchronize the stream

Use --retain-owner as the control case.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

for parent in SCRIPT_PATH.parents:
    if (parent / "physicsnemo" / "__init__.py").exists():
        sys.path.insert(0, str(parent))
        warp_repo = os.environ.get("WARP_REPO")
        warp_path = Path(warp_repo).expanduser() if warp_repo else parent.parent / "warp"
        if (warp_path / "warp" / "__init__.py").exists():
            sys.path.insert(0, str(warp_path))
        break

import torch
import warp as wp

faulthandler.enable()


@wp.kernel
def delay_kernel(out: wp.array(dtype=wp.float32), iterations: wp.int32):
    tid = wp.tid()
    x = wp.float32(tid) * wp.float32(0.000001)
    i = wp.int32(0)
    while i < iterations:
        x = x * wp.float32(1.0000001) + wp.float32(0.000001)
        if x > wp.float32(1024.0):
            x = x * wp.float32(0.5)
        i += 1
    out[tid] = x


@wp.kernel
def mesh_query_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]
    res = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist)
    mesh = wp.mesh_get(mesh_id)

    p0 = mesh.points[mesh.indices[3 * res.face + 0]]
    p1 = mesh.points[mesh.indices[3 * res.face + 1]]
    p2 = mesh.points[mesh.indices[3 * res.face + 2]]
    closest = res.u * p0 + res.v * p1 + (wp.float32(1.0) - res.u - res.v) * p2
    sdf[tid] = res.sign * wp.length(p - closest)


@wp.kernel
def hashgrid_query_kernel(
    grid_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    q = queries[tid]
    query = wp.hash_grid_query(grid_id, q, radius)
    radius_squared = radius * radius
    index = int(0)
    count = wp.int32(0)

    while wp.hash_grid_query_next(query, index):
        delta = q - points[index]
        if wp.dot(delta, delta) <= radius_squared:
            count += 1

    counts[tid] = count


def require_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("This reproducer requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(device)
    wp.init()
    return device


def make_points(count: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return (2.0 * torch.rand((count, 3), device=device, generator=gen) - 1.0).contiguous()


def make_far_points(count: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(count, device=device, dtype=torch.float32)
    points = torch.stack(
        (
            1000.0 + base,
            2000.0 + 0.5 * base,
            3000.0 - 0.25 * base,
        ),
        dim=1,
    )
    return points.contiguous()


def make_plane_mesh(
    resolution: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, resolution + 1, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    vertices = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).reshape(-1, 3)
    vertices = vertices.contiguous()

    cell = torch.arange(resolution * resolution, device=device, dtype=torch.int64)
    row = cell // resolution
    col = cell % resolution
    v00 = row * (resolution + 1) + col
    v10 = v00 + 1
    v01 = v00 + resolution + 1
    v11 = v01 + 1
    faces = torch.cat(
        (
            torch.stack((v00, v10, v11), dim=1),
            torch.stack((v00, v11, v01), dim=1),
        ),
        dim=0,
    )
    return vertices, faces.to(torch.int32).contiguous()


def make_shifted_plane_mesh(
    resolution: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = make_plane_mesh(resolution, device)
    vertices = vertices + torch.tensor((1000.0, 1000.0, 1000.0), device=device)
    return vertices.contiguous(), faces


def enqueue_delay(args: argparse.Namespace, device: torch.device, wp_stream: wp.Stream) -> list[torch.Tensor]:
    buffers = []
    for _ in range(args.delay_launches):
        buf = torch.empty(args.delay_blocks, device=device, dtype=torch.float32)
        wp.launch(
            delay_kernel,
            dim=args.delay_blocks,
            inputs=[wp.from_torch(buf, dtype=wp.float32), int(args.delay_iters)],
            stream=wp_stream,
        )
        buffers.append(buf)
    return buffers


def run_mesh(args: argparse.Namespace) -> None:
    device = require_cuda(args.device)
    stream = torch.cuda.Stream(device=device)
    wp_stream = wp.stream_from_torch(stream)
    pressure_stream = torch.cuda.Stream(device=device)
    wp_pressure_stream = wp.stream_from_torch(pressure_stream)
    vertices, faces = make_plane_mesh(args.mesh_resolution, device)
    replacement_vertices, replacement_faces = make_shifted_plane_mesh(
        args.replacement_mesh_resolution,
        device,
    )
    query_points = make_points(args.queries, device, args.seed)
    expected_sum: float | None = None

    for iteration in range(args.iterations):
        out = torch.empty(args.queries, device=device, dtype=torch.float32)

        with torch.cuda.stream(stream), wp.ScopedStream(wp_stream):
            wp_vertices = wp.from_torch(vertices, dtype=wp.vec3)
            wp_faces = wp.from_torch(faces.reshape(-1), dtype=wp.int32)
            wp_queries = wp.from_torch(query_points, dtype=wp.vec3)
            wp_out = wp.from_torch(out, dtype=wp.float32)

            mesh = wp.Mesh(points=wp_vertices, indices=wp_faces)
            mesh_id = int(mesh.id)
            delay_buffers = enqueue_delay(args, device, wp_stream)
            wp.launch(
                mesh_query_kernel,
                dim=args.queries,
                inputs=[mesh_id, wp_queries, float(args.max_dist), wp_out],
                stream=wp_stream,
            )

        print(
            f"[mesh] iter={iteration} launched query with mesh_id={mesh_id}; "
            f"delay_launches={args.delay_launches} delay_iters={args.delay_iters}",
            flush=True,
        )

        if args.retain_owner:
            print("[mesh] retaining wp.Mesh until synchronize", flush=True)
        else:
            print("[mesh] deleting wp.Mesh immediately after launch", flush=True)
            del mesh
            gc.collect()

        replacement_meshes = []
        if args.post_delete_churn:
            with torch.cuda.stream(pressure_stream), wp.ScopedStream(wp_pressure_stream):
                replacement_wp_vertices = wp.from_torch(replacement_vertices, dtype=wp.vec3)
                replacement_wp_faces = wp.from_torch(
                    replacement_faces.reshape(-1), dtype=wp.int32
                )
                for _ in range(args.post_delete_churn):
                    replacement = wp.Mesh(
                        points=replacement_wp_vertices,
                        indices=replacement_wp_faces,
                    )
                    replacement_meshes.append(replacement)
            reused = sum(int(replacement.id) == mesh_id for replacement in replacement_meshes)
            print(
                f"[mesh] post-delete replacement meshes={len(replacement_meshes)} "
                f"reused_mesh_id={reused}",
                flush=True,
            )
            pressure_stream.synchronize()

        stream.synchronize()
        actual_sum = float(out.sum().item())
        if expected_sum is None:
            expected_sum = actual_sum
        delta = abs(actual_sum - expected_sum)
        print(
            f"[mesh] iter={iteration} survived sum={actual_sum:.6f} "
            f"delta_from_iter0={delta:.6f}",
            flush=True,
        )
        if args.assert_stable and delta > args.tolerance:
            raise RuntimeError(
                f"mesh result changed: expected {expected_sum}, got {actual_sum}"
            )

        if args.retain_owner:
            del mesh
        del replacement_meshes, delay_buffers, wp_vertices, wp_faces, wp_queries, wp_out
        gc.collect()


def run_hashgrid(args: argparse.Namespace) -> None:
    device = require_cuda(args.device)
    stream = torch.cuda.Stream(device=device)
    wp_stream = wp.stream_from_torch(stream)
    pressure_stream = torch.cuda.Stream(device=device)
    wp_pressure_stream = wp.stream_from_torch(pressure_stream)
    points = make_points(args.points, device, args.seed)
    queries = make_points(args.queries, device, args.seed + 1)
    replacement_points = make_far_points(args.replacement_points, device)
    expected_count_sum: int | None = None

    for iteration in range(args.iterations):
        counts = torch.empty(args.queries, device=device, dtype=torch.int32)

        with torch.cuda.stream(stream), wp.ScopedStream(wp_stream):
            wp_points = wp.from_torch(points, dtype=wp.vec3)
            wp_queries = wp.from_torch(queries, dtype=wp.vec3)
            wp_counts = wp.from_torch(counts, dtype=wp.int32)

            grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=wp_points.device)
            grid.reserve(args.points)
            grid.build(points=wp_points, radius=0.5 * float(args.radius))
            grid_id = int(grid.id)

            delay_buffers = enqueue_delay(args, device, wp_stream)
            wp.launch(
                hashgrid_query_kernel,
                dim=args.queries,
                inputs=[grid_id, wp_points, wp_queries, float(args.radius), wp_counts],
                stream=wp_stream,
            )

        print(
            f"[hashgrid] iter={iteration} launched query with grid_id={grid_id}; "
            f"delay_launches={args.delay_launches} delay_iters={args.delay_iters}",
            flush=True,
        )

        if args.retain_owner:
            print("[hashgrid] retaining wp.HashGrid until synchronize", flush=True)
        else:
            print("[hashgrid] deleting wp.HashGrid immediately after launch", flush=True)
            del grid
            gc.collect()

        replacement_grids = []
        if args.post_delete_churn:
            with torch.cuda.stream(pressure_stream), wp.ScopedStream(wp_pressure_stream):
                replacement_wp_points = wp.from_torch(replacement_points, dtype=wp.vec3)
                for _ in range(args.post_delete_churn):
                    replacement = wp.HashGrid(
                        dim_x=128,
                        dim_y=128,
                        dim_z=128,
                        device=replacement_wp_points.device,
                    )
                    replacement.reserve(args.replacement_points)
                    replacement.build(
                        points=replacement_wp_points,
                        radius=float(args.replacement_radius),
                    )
                    replacement_grids.append(replacement)
            reused = sum(int(replacement.id) == grid_id for replacement in replacement_grids)
            print(
                f"[hashgrid] post-delete replacement grids={len(replacement_grids)} "
                f"reused_grid_id={reused}",
                flush=True,
            )
            pressure_stream.synchronize()

        stream.synchronize()
        count_sum = int(counts.sum().item())
        print(
            f"[hashgrid] iter={iteration} survived count_sum={count_sum}",
            flush=True,
        )
        if expected_count_sum is None:
            expected_count_sum = count_sum
        if args.assert_stable and count_sum != expected_count_sum:
            raise RuntimeError(
                f"hashgrid result changed: expected {expected_count_sum}, got {count_sum}"
            )

        if args.retain_owner:
            del grid
        del replacement_grids, delay_buffers, wp_points, wp_queries, wp_counts
        gc.collect()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--delay-blocks", type=int, default=262_144)
    parser.add_argument("--delay-iters", type=int, default=2048)
    parser.add_argument("--delay-launches", type=int, default=1)
    parser.add_argument(
        "--post-delete-churn",
        type=int,
        default=0,
        help=(
            "After deleting the owner, allocate this many replacement objects "
            "on another stream before synchronizing the target stream."
        ),
    )
    parser.add_argument(
        "--retain-owner",
        action="store_true",
        help="Control mode: keep the Warp object alive until after stream synchronization.",
    )
    parser.add_argument(
        "--assert-stable",
        action="store_true",
        help="Raise if results differ from iteration 0.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    mesh = subparsers.add_parser("mesh")
    add_common(mesh)
    mesh.add_argument("--mesh-resolution", type=int, default=96)
    mesh.add_argument("--replacement-mesh-resolution", type=int, default=32)
    mesh.add_argument("--queries", type=int, default=1_000_000)
    mesh.add_argument("--max-dist", type=float, default=1e8)
    mesh.add_argument("--tolerance", type=float, default=1e-3)
    mesh.set_defaults(func=run_mesh)

    hashgrid = subparsers.add_parser("hashgrid")
    add_common(hashgrid)
    hashgrid.add_argument("--points", type=int, default=65_536)
    hashgrid.add_argument("--replacement-points", type=int, default=1024)
    hashgrid.add_argument("--queries", type=int, default=65_536)
    hashgrid.add_argument("--radius", type=float, default=0.05)
    hashgrid.add_argument("--replacement-radius", type=float, default=100.0)
    hashgrid.set_defaults(func=run_hashgrid)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
