# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""
This file contains the implementation of the conv bias fprop, bprop, wgrad, and dgrad
functions using cudnnpy.
"""

import cudnn
import torch


def get_conv_bias_fprop_cache_key(X_gpu, W_gpu, B_gpu, padding, stride):
    """Generate cache key for conv bias forward propagation graph."""
    return (
        tuple(X_gpu.shape),
        tuple(X_gpu.stride()),
        tuple(W_gpu.shape),
        tuple(W_gpu.stride()),
        tuple(B_gpu.shape),
        tuple(B_gpu.stride()),
        B_gpu.dtype,
        X_gpu.dtype,
        W_gpu.dtype,
        padding,
        stride,
        "fprop",
    )


cudnn_handle = cudnn.create_handle()

stream = torch.cuda.current_stream().cuda_stream
cudnn.set_stream(handle=cudnn_handle, stream=stream)


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
@cudnn.graph_cache(key_fn=get_conv_bias_fprop_cache_key)
def create_conv_bias_fprop_graph(X_gpu, W_gpu, B_gpu, padding, stride):
    """Create cuDNN graph for conv bias forward propagation."""
    padding = [padding, padding]
    stride = [stride, stride]
    dilation = [1, 1]

    io_data_type = cudnn.data_type.FLOAT
    intermediate_data_type = cudnn.data_type.FLOAT
    compute_data_type = cudnn.data_type.FLOAT

    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=intermediate_data_type,
        compute_data_type=compute_data_type,
        handle=cudnn_handle,
    )

    X = graph.tensor(
        name="X", dim=X_gpu.size(), stride=X_gpu.stride(), data_type=X_gpu.dtype
    )
    W = graph.tensor(
        name="W", dim=W_gpu.size(), stride=W_gpu.stride(), data_type=W_gpu.dtype
    )
    B = graph.tensor(
        name="B", dim=B_gpu.size(), stride=B_gpu.stride(), data_type=B_gpu.dtype
    )
    conv_output = graph.conv_fprop(
        image=X, weight=W, padding=padding, stride=stride, dilation=dilation
    )
    bias_output = graph.bias(name="bias", input=conv_output, bias=B)
    Y = bias_output
    Y.set_output(True)

    return graph, [X, W, B, Y]


def get_conv_bias_bgrad_cache_key(ygrad_gpu, bias_shape):
    """Generate cache key for conv bias gradient computation graph."""
    return (
        tuple(ygrad_gpu.shape),
        tuple(ygrad_gpu.stride()),
        tuple(bias_shape),
        ygrad_gpu.dtype,
        "bgrad",
    )


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
@cudnn.graph_cache(key_fn=get_conv_bias_bgrad_cache_key)
def create_conv_bias_bgrad_graph(ygrad_gpu, bias_shape):
    """Create cuDNN graph for conv bias gradient computation."""
    io_data_type = cudnn.data_type.FLOAT
    intermediate_data_type = cudnn.data_type.FLOAT
    compute_data_type = cudnn.data_type.FLOAT
    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=intermediate_data_type,
        compute_data_type=compute_data_type,
        handle=cudnn_handle,
    )
    Y_GRAD = graph.tensor(
        name="Y_GRAD",
        dim=ygrad_gpu.size(),
        stride=ygrad_gpu.stride(),
        data_type=ygrad_gpu.dtype,
    )
    bgrad_output = graph.reduction(
        name="B_GRAD", input=Y_GRAD, mode=cudnn.reduction_mode.ADD
    )
    bgrad_output.set_output(True).set_dim(bias_shape).set_data_type(ygrad_gpu.dtype)
    return graph, [Y_GRAD, bgrad_output]


def get_conv_bias_wgrad_cache_key(X_gpu, W_grad_gpu, Y_grad_gpu, padding, stride):
    """Generate cache key for conv weight gradient computation graph."""
    return (
        tuple(X_gpu.shape),
        tuple(X_gpu.stride()),
        tuple(W_grad_gpu.shape),
        tuple(W_grad_gpu.stride()),
        tuple(Y_grad_gpu.shape),
        tuple(Y_grad_gpu.stride()),
        X_gpu.dtype,
        W_grad_gpu.dtype,
        Y_grad_gpu.dtype,
        padding,
        stride,
        "wgrad",
    )


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
@cudnn.graph_cache(key_fn=get_conv_bias_wgrad_cache_key)
def create_conv_bias_wgrad_graph(X_gpu, W_grad_gpu, Y_grad_gpu, padding, stride):
    """Create cuDNN graph for conv weight gradient computation."""
    io_data_type = cudnn.data_type.FLOAT
    intermediate_data_type = cudnn.data_type.FLOAT
    compute_data_type = cudnn.data_type.FLOAT
    padding = [padding, padding]
    stride = [stride, stride]
    dilation = [1, 1]

    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=intermediate_data_type,
        compute_data_type=compute_data_type,
        handle=cudnn_handle,
    )
    X = graph.tensor(
        name="X", dim=X_gpu.size(), stride=X_gpu.stride(), data_type=X_gpu.dtype
    )
    Y_GRAD = graph.tensor(
        name="Y_GRAD",
        dim=Y_grad_gpu.size(),
        stride=Y_grad_gpu.stride(),
        data_type=Y_grad_gpu.dtype,
    )
    wgrad_output = graph.conv_wgrad(
        name="wgrad",
        image=X,
        loss=Y_GRAD,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    wgrad_output.set_output(True).set_dim(W_grad_gpu.size()).set_data_type(
        W_grad_gpu.dtype
    ).set_stride(W_grad_gpu.stride())
    return graph, [X, Y_GRAD, wgrad_output]


def get_conv_bias_dgrad_cache_key(X_gpu, W_gpu, Y_grad_gpu, padding, stride):
    """Generate cache key for conv data gradient computation graph."""
    return (
        tuple(X_gpu.shape),
        tuple(X_gpu.stride()),
        tuple(W_gpu.shape),
        tuple(W_gpu.stride()),
        tuple(Y_grad_gpu.shape),
        tuple(Y_grad_gpu.stride()),
        W_gpu.dtype,
        Y_grad_gpu.dtype,
        padding,
        stride,
        "dgrad",
    )


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
@cudnn.graph_cache(key_fn=get_conv_bias_dgrad_cache_key)
def create_conv_bias_dgrad_graph(X_gpu, W_gpu, Y_grad_gpu, padding, stride):
    """Create cuDNN graph for conv data gradient computation."""
    io_data_type = cudnn.data_type.FLOAT
    intermediate_data_type = cudnn.data_type.FLOAT
    compute_data_type = cudnn.data_type.FLOAT
    padding = [padding, padding]
    stride = [stride, stride]
    dilation = [1, 1]

    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=intermediate_data_type,
        compute_data_type=compute_data_type,
        handle=cudnn_handle,
    )
    W = graph.tensor(
        name="W", dim=W_gpu.size(), stride=W_gpu.stride(), data_type=W_gpu.dtype
    )
    Y_GRAD = graph.tensor(
        name="Y_GRAD",
        dim=Y_grad_gpu.size(),
        stride=Y_grad_gpu.stride(),
        data_type=Y_grad_gpu.dtype,
    )
    dgrad_output = graph.conv_dgrad(
        name="dgrad",
        loss=Y_GRAD,
        filter=W,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    dgrad_output.set_output(True).set_dim(X_gpu.size()).set_data_type(
        X_gpu.dtype
    ).set_stride(X_gpu.stride())
    return graph, [W, Y_GRAD, dgrad_output]


class CudnnGraphManager:
    """Manages cuDNN graph creation and caching for conv bias operations."""

    def __init__(self, handle):
        """Initialize the graph manager with cuDNN handle."""
        self.graphs = {}
        self.graphs["fprop"] = {}
        self.graphs["bgrad"] = {}
        self.graphs["wgrad"] = {}
        self.graphs["dgrad"] = {}
        self.graph_uids = {}
        self.graph_uids["fprop"] = {}
        self.graph_uids["bgrad"] = {}
        self.graph_uids["wgrad"] = {}
        self.graph_uids["dgrad"] = {}
        self.handle = handle
        self.output_buffers_shape = {}
        # returns nothing

    def create_or_get_conv_bias_fprop_graph(self, X_gpu, W_gpu, B_gpu, padding, stride):
        """Create or retrieve conv bias forward propagation graph."""
        key = get_conv_bias_fprop_cache_key(X_gpu, W_gpu, B_gpu, padding, stride)
        if key not in self.graphs["fprop"]:
            graph, uids = create_conv_bias_fprop_graph(
                X_gpu, W_gpu, B_gpu, padding, stride
            )
            self.graphs["fprop"][key] = graph
            self.graph_uids["fprop"][key] = uids
        return self.graphs["fprop"][key], self.graph_uids["fprop"][key]

    def get_graph_uids(self, graph_type, key):
        """Get graph UIDs for specified graph type and key."""
        if key not in self.graph_uids[graph_type]:
            raise KeyError(
                f"Graph uids not found for key: {key}. Available keys: {list(self.graph_uids[graph_type].keys())}"
            )
        return self.graph_uids[graph_type][key]

    def create_or_get_conv_bias_bgrad_graph(self, ygrad_gpu, bias_shape):
        """Create or retrieve conv bias gradient computation graph."""
        key = get_conv_bias_bgrad_cache_key(ygrad_gpu, bias_shape)
        if key not in self.graphs["bgrad"]:
            graph, uids = create_conv_bias_bgrad_graph(ygrad_gpu, bias_shape)
            self.graphs["bgrad"][key] = graph
            self.graph_uids["bgrad"][key] = uids
        return self.graphs["bgrad"][key], self.graph_uids["bgrad"][key]

    def create_or_get_conv_bias_wgrad_graph(
        self, X_gpu, W_grad_gpu, Y_grad_gpu, padding, stride
    ):
        """Create or retrieve conv weight gradient computation graph."""
        key = get_conv_bias_wgrad_cache_key(
            X_gpu, W_grad_gpu, Y_grad_gpu, padding, stride
        )
        if key not in self.graphs["wgrad"]:
            graph, uids = create_conv_bias_wgrad_graph(
                X_gpu, W_grad_gpu, Y_grad_gpu, padding, stride
            )
            self.graphs["wgrad"][key] = graph
            self.graph_uids["wgrad"][key] = uids
        return self.graphs["wgrad"][key], self.graph_uids["wgrad"][key]

    def create_or_get_conv_bias_dgrad_graph(
        self, X_gpu, W_gpu, Y_grad_gpu, padding, stride
    ):
        """Create or retrieve conv data gradient computation graph."""
        key = get_conv_bias_dgrad_cache_key(X_gpu, W_gpu, Y_grad_gpu, padding, stride)
        if key not in self.graphs["dgrad"]:
            graph, uids = create_conv_bias_dgrad_graph(
                X_gpu, W_gpu, Y_grad_gpu, padding, stride
            )
            self.graphs["dgrad"][key] = graph
            self.graph_uids["dgrad"][key] = uids
        return self.graphs["dgrad"][key], self.graph_uids["dgrad"][key]

    def dump_graph(self, graph_type, key):
        """Print graph execution plan count and workspace size."""
        print(
            f"execution plan count: {self.graphs[graph_type][key].get_execution_plan_count()}"
        )
        print(f"workspace size: {self.graphs[graph_type][key].get_workspace_size()}")


cudnn_graph_manager = CudnnGraphManager(cudnn_handle)


def get_fwd_conv_padded_image_dim(tensorDim, pad):
    """Calculate padded image dimension for forward convolution."""
    return tensorDim + (2 * pad)


def get_fwd_conv_dilated_filter_dim(filterDim, dilation):
    """Calculate dilated filter dimension for forward convolution."""
    return ((filterDim - 1) * dilation) + 1


def get_fwd_conv_output_dim(tensorDim, pad, filterDim, stride, dilation):
    """Calculate output dimension for forward convolution."""
    p = (
        int(
            (
                get_fwd_conv_padded_image_dim(tensorDim, pad)
                - get_fwd_conv_dilated_filter_dim(filterDim, dilation)
            )
        )
        / stride
        + 1
    )
    return p


def get_output_buffer_shape(x, weight, bias, padding, stride):
    """Calculate output buffer shape for conv bias operation."""
    return (
        x.shape[0],
        weight.shape[0],
        int(get_fwd_conv_output_dim(x.shape[2], padding, weight.shape[2], stride, 1)),
        int(get_fwd_conv_output_dim(x.shape[3], padding, weight.shape[3], stride, 1)),
    )


def execute_fprop(graph, uids, X_gpu, W_gpu, B_gpu, Y_a, workspace):
    """Execute forward propagation graph."""
    X, W, B, Y = uids
    graph.execute(
        {X: X_gpu, W: W_gpu, B: B_gpu, Y: Y_a}, workspace, handle=cudnn_handle
    )


def execute_bgrad(graph, uids, ygrad_gpu, bgrad_gpu, workspace):
    """Execute bias gradient computation graph."""
    Y_GRAD, B_GRAD = uids
    graph.execute(
        {Y_GRAD: ygrad_gpu, B_GRAD: bgrad_gpu}, workspace, handle=cudnn_handle
    )


def execute_wgrad(graph, uids, x_gpu, wgrad_gpu, ygrad_gpu, workspace):
    """Execute weight gradient computation graph."""
    X, Y_GRAD, W_GRAD = uids
    graph.execute(
        {X: x_gpu, Y_GRAD: ygrad_gpu, W_GRAD: wgrad_gpu}, workspace, handle=cudnn_handle
    )


def execute_dgrad(graph, uids, w_gpu, ygrad_gpu, dgrad_gpu, workspace):
    """Execute data gradient computation graph."""
    W, Y_GRAD, D_GRAD = uids
    graph.execute(
        {W: w_gpu, Y_GRAD: ygrad_gpu, D_GRAD: dgrad_gpu}, workspace, handle=cudnn_handle
    )


def conv_bias_fprop(x, weight, bias, padding, stride):
    """Perform forward propagation for conv bias operation using cuDNN."""
    print("========== using cudnnpy for fprop ==========")
    bias_v = bias.view(1, -1, 1, 1)
    output_shape = get_output_buffer_shape(x, weight, bias, padding, stride)
    print(f"output_shape: {output_shape}")
    Y_actual = torch.empty(
        output_shape, device="cuda", dtype=x.dtype, memory_format=torch.channels_last
    )
    graph, uids = cudnn_graph_manager.create_or_get_conv_bias_fprop_graph(
        x, weight, bias_v, padding, stride
    )
    # allocate workspace
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    execute_fprop(graph, uids, x, weight, bias_v, Y_actual, workspace)

    workspace = None
    return Y_actual


def conv_bias_bprop(x, weight, grad_output, padding, stride):
    """Perform backward propagation for conv bias operation using cuDNN."""
    print("========== using cudnnpy for bprop ==========")
    bias_shape = (1, grad_output.shape[1], 1, 1)
    bgrad_graph, bgrad_uids = cudnn_graph_manager.create_or_get_conv_bias_bgrad_graph(
        grad_output, bias_shape
    )
    bgrad_workspace = torch.empty(
        bgrad_graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    bgrad_output = torch.empty(
        bias_shape,
        device="cuda",
        dtype=grad_output.dtype,
        memory_format=torch.channels_last,
    )
    execute_bgrad(bgrad_graph, bgrad_uids, grad_output, bgrad_output, bgrad_workspace)
    bgrad_workspace = None

    wgrad_graph, wgrad_uids = cudnn_graph_manager.create_or_get_conv_bias_wgrad_graph(
        x, weight, grad_output, padding, stride
    )
    wgrad_workspace = torch.empty(
        wgrad_graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    wgrad_output = torch.empty_like(weight)
    execute_wgrad(
        wgrad_graph, wgrad_uids, x, wgrad_output, grad_output, wgrad_workspace
    )
    wgrad_workspace = None

    dgrad_graph, dgrad_uids = cudnn_graph_manager.create_or_get_conv_bias_dgrad_graph(
        x, weight, grad_output, padding, stride
    )
    dgrad_workspace = torch.empty(
        dgrad_graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    dgrad_output = torch.empty_like(x)
    execute_dgrad(
        dgrad_graph, dgrad_uids, weight, grad_output, dgrad_output, dgrad_workspace
    )
    dgrad_workspace = None

    return dgrad_output, wgrad_output, bgrad_output, None, None
