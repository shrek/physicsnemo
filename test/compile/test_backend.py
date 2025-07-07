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
        example_inputs = [
            torch.randn(1, 9, 1056, 1792, device="cuda", requires_grad=True).to(
                memory_format=torch.channels_last
            ),  # input
        ]
        self.backend_cfg = {
            "enable_conv_bias_fusion": True,
            "amp_mode": False,
        }
        backend = PhysicsNemoBackend(self.backend_cfg)
        compiled_mod = torch.compile(model, backend=backend.backend())
        actual_result = compiled_mod(*example_inputs)
        loss = actual_result.sum()
        loss.backward()


if __name__ == "__main__":
    unittest.main()
