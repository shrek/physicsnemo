#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused async lifetime reproducers for Warp-backed PhysicsNeMo ops.

This script checks whether local Python owners such as ``wp.Mesh`` and
``wp.HashGrid`` are destroyed while CUDA work using their raw ids is still
pending on the launch stream.

Example commands:

  python warp_object_lifetime_repro.py sdf --queries 1000000
  python warp_object_lifetime_repro.py neighbor --points 65536 --queries 65536
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PHYSICSNEMO_ROOT: Path | None = None

for parent in SCRIPT_PATH.parents:
    if (parent / "physicsnemo" / "__init__.py").exists():
        PHYSICSNEMO_ROOT = parent
        sys.path.insert(0, str(parent))
        break

warp_repo = os.environ.get("WARP_REPO")
warp_candidates = []
if warp_repo:
    warp_candidates.append(Path(warp_repo).expanduser())
if PHYSICSNEMO_ROOT is not None:
    warp_candidates.append(PHYSICSNEMO_ROOT.parent / "warp")

for candidate in warp_candidates:
    if (candidate / "warp" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break

import torch
import warp as wp


@dataclass
class DestructorHit:
    label: str
    object_id: int
    warp_id: int | None
    stream_pending: bool | None
    timestamp: float


@contextlib.contextmanager
def traced_destructor(
    cls: type,
    label: str,
    stream: torch.cuda.Stream | None,
    hits: list[DestructorHit],
) -> Iterator[None]:
    """Patch a Warp object's destructor and record whether the stream is busy."""

    original_del = getattr(cls, "__del__", None)

    def _patched_del(self: Any) -> None:
        try:
            pending = None if stream is None else not stream.query()
        except Exception:
            pending = None

        warp_id = getattr(self, "id", None)
        hits.append(
            DestructorHit(
                label=label,
                object_id=id(self),
                warp_id=int(warp_id) if warp_id is not None else None,
                stream_pending=pending,
                timestamp=time.perf_counter(),
            )
        )
        print(
            f"[destructor] {label} object={id(self)} warp_id={warp_id} "
            f"stream_pending={pending}",
            flush=True,
        )

        if original_del is not None:
            original_del(self)

    setattr(cls, "__del__", _patched_del)
    try:
        yield
    finally:
        setattr(cls, "__del__", original_del)


def require_cuda(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("This reproducer requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(dev)
    wp.init()
    return dev


def make_plane_mesh(resolution: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic triangular mesh large enough to stress BVH builds."""

    if resolution < 2:
        raise ValueError("--mesh-resolution must be >= 2")

    coords = torch.linspace(-1.0, 1.0, resolution + 1, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    zz = torch.zeros_like(xx)
    vertices = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).contiguous()

    cell = torch.arange(resolution * resolution, device=device, dtype=torch.int64)
    row = cell // resolution
    col = cell % resolution
    v00 = row * (resolution + 1) + col
    v10 = v00 + 1
    v01 = v00 + (resolution + 1)
    v11 = v01 + 1

    tri0 = torch.stack((v00, v10, v11), dim=1)
    tri1 = torch.stack((v00, v11, v01), dim=1)
    faces = torch.cat((tri0, tri1), dim=0).to(torch.int32).contiguous()
    return vertices, faces


def make_points(count: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return (2.0 * torch.rand((count, 3), device=device, generator=gen) - 1.0).contiguous()


def print_lifetime_summary(hits: list[DestructorHit], stream: torch.cuda.Stream) -> None:
    gc.collect()
    print(f"[summary] destructor_hits={len(hits)} stream_pending_now={not stream.query()}")
    for hit in hits:
        print(
            "[summary] "
            f"{hit.label} object={hit.object_id} warp_id={hit.warp_id} "
            f"stream_pending_at_del={hit.stream_pending}"
        )


def run_sdf(args: argparse.Namespace) -> None:
    from physicsnemo.nn.functional.geometry.sdf import signed_distance_field

    device = require_cuda(args.device)
    stream = torch.cuda.Stream(device=device)
    vertices, faces = make_plane_mesh(args.mesh_resolution, device)
    queries = make_points(args.queries, device, args.seed)

    torch.cuda.synchronize(device)
    hits: list[DestructorHit] = []
    with traced_destructor(wp.Mesh, "wp.Mesh", stream, hits):
        with torch.cuda.stream(stream):
            sdf, hit_points = signed_distance_field(
                vertices,
                faces,
                queries,
                use_sign_winding_number=args.use_sign_winding_number,
            )
            keep_alive = (sdf, hit_points)

        print_lifetime_summary(hits, stream)
        stream.synchronize()
        print(f"[done] sdf_sum={float(keep_alive[0].float().sum().item()):.6f}")


def run_neighbor(args: argparse.Namespace) -> None:
    from physicsnemo.nn.functional.neighbors.radius_search._warp_impl import (
        radius_search_impl,
    )

    device = require_cuda(args.device)
    stream = torch.cuda.Stream(device=device)
    points = make_points(args.points, device, args.seed).unsqueeze(0)
    queries = make_points(args.queries, device, args.seed + 1).unsqueeze(0)

    torch.cuda.synchronize(device)
    hits: list[DestructorHit] = []
    with traced_destructor(wp.HashGrid, "wp.HashGrid", stream, hits):
        with torch.cuda.stream(stream):
            outputs = radius_search_impl(
                points,
                queries,
                float(args.radius),
                int(args.max_points),
                bool(args.return_dists),
                bool(args.return_points),
            )
            keep_alive = outputs

        print_lifetime_summary(hits, stream)
        stream.synchronize()
        print(f"[done] indices_shape={tuple(keep_alive[0].shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--seed", type=int, default=1234)
        subparser.add_argument("--radius", type=float, default=0.05)

    sdf = subparsers.add_parser("sdf")
    add_common(sdf)
    sdf.add_argument("--mesh-resolution", type=int, default=96)
    sdf.add_argument("--queries", type=int, default=1_000_000)
    sdf.add_argument("--use-sign-winding-number", action="store_true")
    sdf.set_defaults(func=run_sdf)

    neighbor = subparsers.add_parser("neighbor")
    add_common(neighbor)
    neighbor.add_argument("--points", type=int, default=65_536)
    neighbor.add_argument("--queries", type=int, default=65_536)
    neighbor.add_argument("--max-points", type=int, default=8)
    neighbor.add_argument("--return-dists", action="store_true")
    neighbor.add_argument("--return-points", action="store_true")
    neighbor.set_defaults(func=run_neighbor)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
