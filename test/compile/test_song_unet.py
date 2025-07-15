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
"""Test song unet with custom backend"""
import copy

import pytest
import torch

from physicsnemo.models.diffusion import SongUNet as UNet


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_backend(device):
    """Test song unet with custom backend"""

    def compare_conv2d_gradients(model1, model2):
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            # print(f"checking {name1}")
            if param1.grad is None or param2.grad is None:
                continue
            torch.testing.assert_close(param1.grad, param2.grad, atol=0.05, rtol=1e-2)

    def setup_model():
        """Setup the model for testing"""
        model = (
            UNet(
                img_resolution=16,
                in_channels=2,
                out_channels=2,
                embedding_type="fourier",
                channel_mult_noise=2,
                encoder_type="residual",
                resample_filter=[1, 3, 3, 1],
                use_apex_gn=True,
                amp_mode=False,
            )
            .to(device)
            .to(memory_format=torch.channels_last)
        )
        noise_labels = torch.randn([1]).to(device).to(dtype=torch.float32)
        class_labels = torch.randint(0, 1, (1, 1)).to(device).to(dtype=torch.float32)
        input_image = (
            torch.randn([1, 2, 448, 448])
            .to(device)
            .to(dtype=torch.float32)
            .to(memory_format=torch.channels_last)
        )

        return model, [input_image, noise_labels, class_labels]

    # Check fullgraph compilation
    # run only on GPU
    from physicsnemo.compile.backend import PhysicsNemoBackend

    model, invar = setup_model()
    model_ = copy.deepcopy(model)
    invar_ = copy.deepcopy(invar)

    backend_cfg = {
        "enable_conv_bias_fusion": True,
        "amp_mode": False,
    }
    backend = PhysicsNemoBackend(backend_cfg)
    compiled_mod = torch.compile(model, backend=backend.backend(), fullgraph=True)
    actual_result = compiled_mod(*invar)
    expected_result = model_(*invar_)
    torch.testing.assert_close(expected_result, actual_result, atol=0.05, rtol=1e-2)
    print("expected and actual results are close, Forward pass successful!")

    loss = actual_result.sum()
    loss.backward()

    loss_ = expected_result.sum()
    loss_.backward()

    torch.testing.assert_close(invar[0].grad, invar_[0].grad, atol=0.05, rtol=1e-2)
    compare_conv2d_gradients(model, model_)
    print("expected and actual gradients are close, Backward pass successful!")


if __name__ == "__main__":
    test_song_unet_backend(device="cuda:0")
