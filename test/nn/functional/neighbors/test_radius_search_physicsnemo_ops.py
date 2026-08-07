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

"""Tests for the optional physicsnemo-ops radius-search adapter."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("physicsnemo_ops")

from physicsnemo_ops.torch import radius_search as primitive_radius_search

from physicsnemo.nn.functional import radius_search
from physicsnemo.nn.functional.neighbors.radius_search import RadiusSearch
from physicsnemo.nn.module.ball_query import BQWarp


def _problem(device: str, batched: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    queries = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [8.0, 8.0, 8.0]],
        dtype=torch.float32,
        device=device,
    )
    if batched:
        points = torch.stack((points, points + 10.0))
        queries = torch.stack((queries, queries + 10.0))
    return points, queries


def test_radius_search_physicsnemo_ops_is_registered() -> None:
    assert "physicsnemo_ops" in RadiusSearch.implementations()
    assert "physicsnemo_ops" in RadiusSearch.available_implementations()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("return_dists", [False, True])
@pytest.mark.parametrize("return_points", [False, True])
def test_radius_search_physicsnemo_ops_return_contract(
    device: str,
    batched: bool,
    return_dists: bool,
    return_points: bool,
) -> None:
    points, queries = _problem(device, batched)
    max_points = points.shape[-2]
    result = radius_search(
        points,
        queries,
        0.5,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation="physicsnemo_ops",
    )
    if return_points and return_dists:
        indices, selected, distances = result
    elif return_points:
        indices, selected = result
        distances = None
    elif return_dists:
        indices, distances = result
        selected = None
    else:
        indices = result
        selected = None
        distances = None

    primitive_indices, counts = primitive_radius_search(
        points.float(), queries.float(), 0.5, max_points
    )
    points_b = points.unsqueeze(0) if points.ndim == 2 else points
    queries_b = queries.unsqueeze(0) if queries.ndim == 2 else queries
    indices_b = indices.unsqueeze(0) if indices.ndim == 2 else indices
    primitive_b = (
        primitive_indices.unsqueeze(0)
        if primitive_indices.ndim == 2
        else primitive_indices
    )
    counts_b = counts.unsqueeze(0) if counts.ndim == 1 else counts
    selected_b = (
        selected.unsqueeze(0)
        if selected is not None and selected.ndim == 3
        else selected
    )
    distances_b = (
        distances.unsqueeze(0)
        if distances is not None and distances.ndim == 2
        else distances
    )

    for batch in range(points_b.shape[0]):
        for query in range(queries_b.shape[1]):
            count = int(counts_b[batch, query])
            actual_ids = indices_b[batch, query, :count].sort().values
            primitive_ids = primitive_b[batch, query, :count].sort().values
            torch.testing.assert_close(actual_ids, primitive_ids)
            assert torch.all(indices_b[batch, query, count:] == 0)
            if selected_b is not None:
                ids = indices_b[batch, query, :count].long()
                torch.testing.assert_close(
                    selected_b[batch, query, :count],
                    points_b[batch, ids],
                )
                assert torch.all(selected_b[batch, query, count:] == 0)
            if distances_b is not None:
                ids = indices_b[batch, query, :count].long()
                expected = torch.linalg.vector_norm(
                    points_b[batch, ids].float() - queries_b[batch, query].float(),
                    dim=-1,
                )
                torch.testing.assert_close(distances_b[batch, query, :count], expected)
                assert torch.all(distances_b[batch, query, count:] == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_radius_search_physicsnemo_ops_precision_and_backward(
    device: str, dtype: torch.dtype
) -> None:
    points, queries = _problem(device)
    points = points.to(dtype).requires_grad_(True)
    queries = queries.to(dtype).requires_grad_(True)
    _, selected = radius_search(
        points,
        queries,
        0.5,
        max_points=points.shape[0],
        return_points=True,
        implementation="physicsnemo_ops",
    )
    assert selected.dtype == dtype
    selected.float().square().sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert queries.grad is None or torch.all(queries.grad == 0)


def test_radius_search_physicsnemo_ops_empty_and_validation(device: str) -> None:
    empty_points = torch.empty((0, 3), device=device)
    queries = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    indices, selected = radius_search(
        empty_points,
        queries,
        0.5,
        max_points=4,
        return_points=True,
        implementation="physicsnemo_ops",
    )
    assert indices.shape == (1, 4)
    assert torch.all(indices == 0)
    assert torch.all(selected == 0)

    with pytest.raises(ValueError, match="requires max_points"):
        radius_search(
            queries,
            queries,
            0.5,
            implementation="physicsnemo_ops",
        )


def test_radius_search_physicsnemo_ops_same_tensor(device: str) -> None:
    points, _ = _problem(device)
    indices, selected = radius_search(
        points,
        points,
        0.75,
        max_points=points.shape[0],
        return_points=True,
        implementation="physicsnemo_ops",
    )
    assert indices.shape == (points.shape[0], points.shape[0])
    for query in range(points.shape[0]):
        valid = (
            torch.linalg.vector_norm(selected[query] - points[query], dim=-1) <= 0.75
        )
        assert valid.any()


def test_radius_search_physicsnemo_ops_compile_and_bqwarp(device: str) -> None:
    if device == "cpu":
        pytest.skip("GeoTransolver compile path is CUDA-only")

    points, queries = _problem(device, batched=True)

    def search_fn(points: torch.Tensor, queries: torch.Tensor):
        return radius_search(
            points,
            queries,
            0.5,
            max_points=points.shape[-2],
            implementation="physicsnemo_ops",
        )

    eager = search_fn(points, queries)
    compiled = torch.compile(search_fn, fullgraph=True)(points, queries)
    torch.testing.assert_close(
        eager.sort(dim=-1).values,
        compiled.sort(dim=-1).values,
    )

    bq = BQWarp(
        radius=0.5,
        neighbors_in_radius=points.shape[-2],
        implementation="physicsnemo_ops",
    )
    compiled_bq = torch.compile(bq, fullgraph=True)
    mapping, selected = compiled_bq(points, queries)
    assert mapping.shape == (points.shape[0], queries.shape[1], points.shape[1])
    assert selected.shape == (*mapping.shape, 3)
