#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused Warp native thread-race reproducers.

This script isolates concurrent entry into Warp spatial-index construction and
destruction paths. Use ``--sync-before-destroy`` to remove the async lifetime
hazard, and compare unlocked runs against ``--lock`` to test whether native
Warp contention is the remaining failure source.

Example commands:

  python warp_thread_race_repro.py mesh --workers 4 --iterations 100
  python warp_thread_race_repro.py hashgrid --workers 4 --iterations 100
  python warp_thread_race_repro.py mixed --workers 4 --iterations 100 --sync-before-destroy
  python warp_thread_race_repro.py mixed --workers 4 --iterations 100 --sync-before-destroy --lock
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
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


def require_cuda(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("This reproducer requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(dev)
    wp.init()
    return dev


def set_thread_cuda_device(device: str) -> torch.device:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
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


@contextlib.contextmanager
def optional_lock(lock: threading.Lock | None) -> Iterator[None]:
    if lock is None:
        yield
    else:
        with lock:
            yield


def mesh_worker(
    args: argparse.Namespace,
    worker_id: int,
    barrier: threading.Barrier,
    setup_lock: threading.Lock,
    lock: threading.Lock | None,
) -> None:
    with setup_lock:
        device = set_thread_cuda_device(args.device)
        stream = torch.cuda.Stream(device=device)
        vertices, faces = make_plane_mesh(args.mesh_resolution, device)
        wp_vertices = wp.from_torch(vertices, dtype=wp.vec3)
        wp_faces = wp.from_torch(faces.reshape(-1), dtype=wp.int32)
        wp_stream = wp.stream_from_torch(stream)
        if args.progress_every:
            print(f"[worker {worker_id}] mesh setup complete", flush=True)

    barrier.wait()
    for step in range(args.iterations):
        with torch.cuda.stream(stream):
            with optional_lock(lock):
                with wp.ScopedStream(wp_stream):
                    mesh = wp.Mesh(
                        points=wp_vertices,
                        indices=wp_faces,
                        support_winding_number=args.use_sign_winding_number,
                    )
                    if args.sync_before_destroy:
                        wp.synchronize_stream(wp_stream)
                    del mesh
        if args.progress_every and step % args.progress_every == 0:
            print(f"[worker {worker_id}] mesh step={step}", flush=True)

    stream.synchronize()


def hashgrid_worker(
    args: argparse.Namespace,
    worker_id: int,
    barrier: threading.Barrier,
    setup_lock: threading.Lock,
    lock: threading.Lock | None,
) -> None:
    with setup_lock:
        device = set_thread_cuda_device(args.device)
        stream = torch.cuda.Stream(device=device)
        points = make_points(args.points, device, args.seed + worker_id)
        wp_points = wp.from_torch(points, dtype=wp.vec3)
        wp_stream = wp.stream_from_torch(stream)
        if args.progress_every:
            print(f"[worker {worker_id}] hashgrid setup complete", flush=True)

    barrier.wait()
    for step in range(args.iterations):
        with torch.cuda.stream(stream):
            with optional_lock(lock):
                with wp.ScopedStream(wp_stream):
                    grid = wp.HashGrid(
                        dim_x=args.grid_dim,
                        dim_y=args.grid_dim,
                        dim_z=args.grid_dim,
                        device=wp_points.device,
                    )
                    grid.reserve(args.points)
                    grid.build(points=wp_points, radius=float(args.radius))
                    if args.sync_before_destroy:
                        wp.synchronize_stream(wp_stream)
                    del grid
        if args.progress_every and step % args.progress_every == 0:
            print(f"[worker {worker_id}] hashgrid step={step}", flush=True)

    stream.synchronize()


def run(args: argparse.Namespace) -> None:
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        with contextlib.suppress(RuntimeError):
            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

    require_cuda(args.device)
    lock = threading.Lock() if args.lock else None
    setup_lock = threading.Lock()
    barrier = threading.Barrier(args.workers)
    threads: list[threading.Thread] = []
    exceptions: list[BaseException] = []
    exceptions_lock = threading.Lock()

    if args.lock and not args.sync_before_destroy:
        print(
            "[warning] --lock serializes native entry, but without "
            "--sync-before-destroy async lifetime remains intentionally exposed.",
            flush=True,
        )

    for worker_id in range(args.workers):
        if args.mode == "mesh":
            target = mesh_worker
        elif args.mode == "hashgrid":
            target = hashgrid_worker
        elif worker_id % 2 == 0:
            target = mesh_worker
        else:
            target = hashgrid_worker

        def wrapped_target(
            worker_target: Any = target,
            current_worker_id: int = worker_id,
        ) -> None:
            try:
                worker_target(args, current_worker_id, barrier, setup_lock, lock)
            except BaseException as exc:
                with exceptions_lock:
                    exceptions.append(exc)
                with contextlib.suppress(Exception):
                    barrier.abort()
                raise

        thread = threading.Thread(
            target=wrapped_target,
            name=f"warp-thread-race-{worker_id}",
        )
        thread.start()
        threads.append(thread)

    try:
        if args.join_timeout <= 0:
            for thread in threads:
                thread.join()
        else:
            deadline = time.monotonic() + args.join_timeout
            for thread in threads:
                while thread.is_alive():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                        raise TimeoutError(
                            f"timed out after {args.join_timeout}s waiting for worker threads"
                        )
                    thread.join(timeout=min(1.0, remaining))
    except KeyboardInterrupt:
        print("[interrupt] dumping all Python thread stacks", file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        raise

    if exceptions:
        raise RuntimeError(
            f"{len(exceptions)} worker thread(s) failed; first error: {exceptions[0]!r}"
        )

    torch.cuda.synchronize(torch.device(args.device))
    print("[done] thread-race run completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mesh", "hashgrid", "mixed"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mesh-resolution", type=int, default=64)
    parser.add_argument("--points", type=int, default=65_536)
    parser.add_argument("--grid-dim", type=int, default=128)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--sync-before-destroy", action="store_true")
    parser.add_argument("--use-sign-winding-number", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--join-timeout",
        type=float,
        default=0.0,
        help="Dump all thread stacks and fail if workers do not finish within this many seconds.",
    )
    parser.set_defaults(func=run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
