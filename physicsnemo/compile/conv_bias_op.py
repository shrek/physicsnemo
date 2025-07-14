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
functions using torch.library.custom_op.
"""

import torch

import physicsnemo.compile.conv_bias_py_impl as conv_bias_py_impl


@torch.library.custom_op(
    "physicsnemo::conv_bias_fprop", mutates_args=(), device_types="cuda"
)
def conv_bias_fprop(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    padding: int,
    stride: int,
) -> torch.Tensor:
    """
    Custom fused convolution with bias forward pass operation.

    This function performs a 2D convolution with bias in a single fused operation
    for improved performance. It automatically converts tensors to channels_last
    memory format for optimal GPU performance and delegates the actual computation
    to the C++ conv_bias implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weight (torch.Tensor): Convolution kernel of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias (torch.Tensor): Bias tensor of shape (out_channels,)
        padding (int): Padding size for the convolution operation
        stride (int): Stride for the convolution operation

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, output_height, output_width)

    Note:
        This function automatically converts input tensors to channels_last memory format
        for optimal performance on GPU. The dilation is fixed to 1.
    """

    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)
    if not weight.is_contiguous(memory_format=torch.channels_last):
        weight = weight.to(memory_format=torch.channels_last)

    out = conv_bias_py_impl.conv_bias_fprop(x, weight, bias, padding, stride)
    return out


@conv_bias_fprop.register_fake
def _(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int, stride: int
) -> torch.Tensor:
    """
    Fake implementation for conv_bias_fprop used during graph tracing.

    This function provides a reference implementation using PyTorch's standard
    conv2d operation for graph tracing and shape inference. It mimics the behavior
    of the actual fused operation but uses the standard PyTorch implementation.

    Args:
        x (torch.Tensor): Input tensor
        weight (torch.Tensor): Convolution kernel
        bias (torch.Tensor): Bias tensor
        padding (int): Padding size
        stride (int): Stride size

    Returns:
        torch.Tensor: Output tensor computed using standard conv2d
    """
    # sanity check
    # print("========== in register_fake for fprop ==========")
    # print(f"x: {x.shape}, weight: {weight.shape}, bias: {bias.shape}")
    # print(f"x: {x.dtype}, weight: {weight.dtype}, bias: {bias.dtype}")
    # print(f"x: {x.dtype}, weight: {weight.dtype}, bias: {bias.dtype}")
    # print(f"x: {x.is_contiguous(memory_format=torch.channels_last)}, weight: {weight.is_contiguous(memory_format=torch.channels_last)}")
    if not x.is_contiguous(memory_format=torch.channels_last):
        # print("---converting input to channels last---")
        x = x.to(memory_format=torch.channels_last)
    if not weight.is_contiguous(memory_format=torch.channels_last):
        # print("---converting weight to channels last---")
        weight = weight.to(memory_format=torch.channels_last)

    # print(f"x: {x.is_contiguous(memory_format=torch.channels_last)}, weight: {weight.is_contiguous(memory_format=torch.channels_last)}")
    out = torch.nn.functional.conv2d(
        x, weight, bias=bias, padding=padding, stride=stride, dilation=1
    )
    return out


@torch.library.custom_op(
    "physicsnemo::conv_bias_bprop", mutates_args=(), device_types="cuda"
)
def conv_bias_bprop(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    padding: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom fused convolution with bias backward pass operation.

    This function computes gradients for the fused convolution operation.
    It delegates the gradient computation to the C++ conv_bias implementation
    and returns gradients with respect to input, weight, and bias.

    Args:
        x (torch.Tensor): Input tensor from forward pass
        weight (torch.Tensor): Convolution kernel from forward pass
        grad_output (torch.Tensor): Gradient of the output
        padding (int): Padding size used in forward pass
        stride (int): Stride used in forward pass

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Gradients with respect to (x, weight, bias, None, None)
    """
    dgrad, wgrad, bgrad, _, _ = conv_bias_py_impl.conv_bias_bprop(
        x, weight, grad_output, padding, stride
    )
    return dgrad, wgrad, bgrad.squeeze(), None, None


@conv_bias_bprop.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    padding: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for conv_bias_bprop used during graph tracing.

    This function provides a reference implementation for gradient computation
    during graph tracing. It creates zero tensors with appropriate shapes
    and memory formats to mimic the actual backward pass.

    Args:
        x (torch.Tensor): Input tensor
        weight (torch.Tensor): Convolution kernel
        grad_output (torch.Tensor): Gradient of the output
        padding (int): Padding size
        stride (int): Stride size

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Zero gradients with appropriate shapes and memory formats
    """
    # print("========== in register_fake for bprop ==========")
    # print(f"grad_output: {grad_output.shape}, x: {x.shape}, weight: {weight.shape}")
    # print(f"grad_output: {grad_output.dtype}, x: {x.dtype}, weight: {weight.dtype}")

    sum = torch.sum(grad_output, dim=[0, 2, 3])
    x_grad = torch.zeros_like(x)
    weight_grad = torch.zeros_like(weight)
    # print(f"x_grad: {x_grad.is_contiguous(memory_format=torch.channels_last)}, weight_grad: {weight_grad.is_contiguous(memory_format=torch.channels_last)}")
    if not x.is_contiguous(memory_format=torch.channels_last):
        # print("---converting x to channels last---")
        x_grad = x_grad.to(memory_format=torch.channels_last)
    if not weight.is_contiguous(memory_format=torch.channels_last):
        # print("---converting weight to channels last---")
        weight_grad = weight_grad.to(memory_format=torch.channels_last)
    if not sum.is_contiguous():
        sum = sum.contiguous()
    # print(f"x_grad: {x_grad.is_contiguous(memory_format=torch.channels_last)}, weight_grad: {weight_grad.is_contiguous(memory_format=torch.channels_last)}")
    bias_grad = torch.zeros_like(sum)
    bias_grad = bias_grad.to(torch.float32)
    # print(f"x_grad: {x_grad.shape}, weight_grad: {weight_grad.shape}, bias_grad: {bias_grad.shape}")
    # print(f"========== x_type: {x.dtype}, weight_type: {weight.dtype}, bias_type: {bias_grad.dtype} ==========")
    # print("========== done register_fake for bprop ==========")
    return x_grad, weight_grad, bias_grad, None, None


def backward(ctx, grad_output):
    """
    Backward pass for the custom conv_bias_fprop operation.

    This function is called during automatic differentiation to compute
    gradients with respect to the inputs of the forward pass. It handles
    memory format conversion and delegates to the backward operation.

    Args:
        ctx: Context object containing saved tensors and parameters
        grad_output (torch.Tensor): Gradient of the output

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
            Gradients with respect to (x, weight, bias, None, None)
    """
    # print("@@@@@@@@@@@@@@@ in backward @@@@@@@@@@@@@@@@")
    x, weight = ctx.saved_tensors
    # print(f"x: {x.is_contiguous(memory_format=torch.channels_last)}, weight: {weight.is_contiguous(memory_format=torch.channels_last)}, \
    #      grad_output: {grad_output.is_contiguous(memory_format=torch.channels_last)}")

    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)
    if not weight.is_contiguous(memory_format=torch.channels_last):
        weight = weight.to(memory_format=torch.channels_last)
    if not grad_output.is_contiguous(memory_format=torch.channels_last):
        grad_output = grad_output.to(memory_format=torch.channels_last)
    # print(f"x: {x.is_contiguous(memory_format=torch.channels_last)}, weight: {weight.is_contiguous(memory_format=torch.channels_last)}, \
    #    grad_output: {grad_output.is_contiguous(memory_format=torch.channels_last)}")
    dx, dw, db, _, _ = conv_bias_bprop(x, weight, grad_output, ctx.padding, ctx.stride)

    print(f"dtype of dx, dw, db: {dx.dtype}, {dw.dtype}, {db.dtype}")
    return dx, dw, db, None, None


def setup_context(ctx, inputs, output):
    """
    Setup context for the custom conv_bias_fprop operation.

    This function is called during the forward pass to save tensors and
    parameters needed for the backward pass. It stores the input tensors
    and operation parameters in the context object.

    Args:
        ctx: Context object to store information for backward pass
        inputs: Tuple of input tensors (x, weight, bias, padding, stride)
        output: Output tensor from forward pass (not used)
    """
    # print("========== in setup_context ==========")
    # print(f"inputs: {inputs}")
    x, weight, bias, padding, stride = inputs
    # print(f"x: {x.shape}, weight: {weight.shape}, bias: {bias.shape}")
    # save for backward
    ctx.save_for_backward(x, weight)
    ctx.padding = padding
    ctx.stride = stride
    ctx.bias_size = bias.shape[0]
    # print("========== done setup_context ==========")


torch.library.register_autograd(
    "physicsnemo::conv_bias_fprop", backward, setup_context=setup_context
)
