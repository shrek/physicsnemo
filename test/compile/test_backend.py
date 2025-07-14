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
"""Test backend"""
import copy
import unittest

import torch

from physicsnemo.compile.backend import PhysicsNemoBackend


class Conv2d(torch.nn.Module):
    """Test conv2d module"""

    def __init__(self, in_channels, out_channels, kernel_size):
        """Initialize conv2d module"""
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=True,
            padding=1,
            stride=1,
            dilation=1,
        )

    def forward(self, x):
        """Forward pass"""
        bias = self.conv.bias
        weight = self.conv.weight
        return torch.nn.functional.conv2d(
            x, weight, bias=bias, padding=1, stride=1, dilation=1
        )


class TestBackend(unittest.TestCase):
    """Test backend"""

    def setUp(self):
        """Set up test backend"""
        super().setUp()

    def test_backend(self):
        """Test backend"""
        model = Conv2d(9, 9, 3).eval().to("cuda").to(memory_format=torch.channels_last)
        model_ = copy.deepcopy(model)
        model__ = copy.deepcopy(model)
        example_input = torch.randn(
            1, 9, 1056, 1792, device="cuda", requires_grad=True
        ).to(
            memory_format=torch.channels_last
        )  # input
        example_input_ = example_input.clone()
        example_input__ = example_input.clone()
        self.backend_cfg = {
            "enable_conv_bias_fusion": True,
            "amp_mode": False,
        }
        backend = PhysicsNemoBackend(self.backend_cfg)
        custom_compiled_model = torch.compile(model, backend=backend.backend())
        actual_result = custom_compiled_model(example_input)
        loss = actual_result.sum()
        loss.backward()

        # test accurach against torch.compile
        compiled_model = torch.compile(model_)
        expected_result = compiled_model(example_input_)
        loss_ = expected_result.sum()
        loss_.backward()

        torch.testing.assert_close(
            actual_result, expected_result, atol=1e-3, rtol=1e-3, equal_nan=True
        )
        torch.testing.assert_close(
            model.conv.weight.grad,
            model_.conv.weight.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        torch.testing.assert_close(
            model.conv.bias.grad,
            model_.conv.bias.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        torch.testing.assert_close(
            example_input.grad,
            example_input_.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )

        # test accuracy against eager mode
        eager_result = model__(example_input__)
        loss__ = eager_result.sum()
        loss__.backward()
        torch.testing.assert_close(
            actual_result, eager_result, atol=1e-3, rtol=1e-3, equal_nan=True
        )
        torch.testing.assert_close(
            model__.conv.weight.grad,
            model.conv.weight.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        torch.testing.assert_close(
            model__.conv.bias.grad,
            model.conv.bias.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        torch.testing.assert_close(
            example_input__.grad,
            example_input.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
