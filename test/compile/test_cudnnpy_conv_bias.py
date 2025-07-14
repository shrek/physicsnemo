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

"""Test conv bias using cudnnpy"""
import copy
import math
import random
import unittest

import torch

import physicsnemo.compile.conv_bias_py_impl as conv_bias_py_impl


class TestConvBias(unittest.TestCase):
    """Test conv bias"""

    def setUp(self, seed=0):
        """Set up test conv bias"""
        super().setUp()
        torch.manual_seed(seed)

        self.batch_size = random.randint(1, 64)
        self.in_channels = random.randint(1, 64) * 8
        self.out_channels = random.randint(1, 64) * 8
        self.in_height = self.in_width = random.randint(5, 100)
        self.conv_kernel_size = random.randint(1, 5)
        self.conv_pad = random.randint(0, int(self.conv_kernel_size / 2))
        self.conv_stride = random.randint(1, 5)
        self.conv_dilation = 1
        self.out_height = self.out_width = math.floor(
            (
                self.in_height
                + 2 * self.conv_pad
                - self.conv_dilation * (self.conv_kernel_size - 1)
                - 1
            )
            / self.conv_stride
            + 1
        )

        self.x = (
            torch.randint(
                low=-16,
                high=16,
                size=[self.batch_size, self.in_channels, self.in_height, self.in_width],
            )
            .cuda()
            .to(memory_format=torch.channels_last)
            .to(dtype=torch.float32)
        )
        self.x_ = self.x.clone()
        self.x.requires_grad_()
        self.x_.requires_grad_()

        self.bias = torch.randn([1, self.out_channels, 1, 1]).cuda()
        self.bias_ = self.bias.clone()

        self.conv1 = (
            torch.nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_pad,
            )
            .cuda()
            .to(memory_format=torch.channels_last)
            .to(dtype=torch.float32)
        )
        self.conv1_ = copy.deepcopy(self.conv1)

        print()
        print(
            "> input=[{}, {}, {}, {}]".format(
                self.batch_size, self.in_channels, self.in_height, self.in_width
            )
        )
        print(
            "> kernel=[{}, {}, {}, {}], stride={}, pad={}".format(
                self.out_channels,
                self.in_channels,
                self.conv_kernel_size,
                self.conv_kernel_size,
                self.conv_stride,
                self.conv_pad,
            )
        )

    def test_conv_bias(self):
        """Test conv bias"""
        weight = self.conv1.weight.to(memory_format=torch.channels_last).to(
            dtype=torch.float32
        )
        bias = (
            self.conv1.bias.reshape(1, -1, 1, 1)
            .to(memory_format=torch.channels_last)
            .to(dtype=torch.float32)
        )
        out = conv_bias_py_impl.conv_bias_fprop(
            self.x, weight, bias, self.conv_pad, self.conv_stride
        )
        out_ = self.conv1_(self.x_)
        res = out[0]
        res = res.requires_grad_(True)
        loss = res.sum()
        loss.backward()
        torch.testing.assert_close(out, out_, atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(
            self.conv1_.bias.grad,
            self.conv1.bias.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        torch.testing.assert_close(
            self.conv1_.weight.grad,
            self.conv1.weight.grad,
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        torch.testing.assert_close(
            self.x_.grad, self.x.grad, atol=1e-3, rtol=1e-3, equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
