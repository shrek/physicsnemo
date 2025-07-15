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

"""Test conv bias using cudnn python api"""
import copy
import unittest

import torch

import physicsnemo.compile.conv_bias_py_impl as conv_bias_py_impl


class TestConvBias(unittest.TestCase):
    """Test conv bias"""

    def setUp(self, seed=0):
        """Set up test conv bias"""
        super().setUp()
        torch.manual_seed(seed)

        self.batch_size = 1
        self.in_channels = 8
        self.out_channels = 8
        self.in_height = 1056
        self.in_width = 1972
        self.conv_kernel_size = 3
        self.conv_pad = 1
        self.conv_stride = 1
        self.conv_dilation = 1

        self.x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.in_height,
            self.in_width,
            device="cuda",
            requires_grad=True,
        )
        self.x_ = self.x.clone()
        self.x.to(memory_format=torch.channels_last)
        self.x_.to(memory_format=torch.channels_last)

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

        print("========== forward test ==========")
        print("checking output")
        torch.testing.assert_close(out, out_, atol=1e-3, rtol=1e-3, equal_nan=True)

        expected_grads = torch.ops.aten.convolution_backward(
            out_,
            self.x,
            self.conv1_.weight,
            self.bias.shape,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
        actual_grads = conv_bias_py_impl.conv_bias_bprop(
            self.x, self.conv1.weight, out_, 1, 1
        )

        print("========== backward test ==========")
        print("checking input grads")
        torch.testing.assert_close(
            expected_grads[0],
            actual_grads[0],
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        print("checking weight grads")
        torch.testing.assert_close(
            expected_grads[1],
            actual_grads[1],
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )
        print("checking bias grads")
        torch.testing.assert_close(
            expected_grads[1],
            actual_grads[1],
            atol=1e-3,
            rtol=1e-3,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
