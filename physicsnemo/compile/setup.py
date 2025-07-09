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
"""Setup script for building the custom ops"""
import os
import shutil
import subprocess
import tempfile

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Create a temporary directory for cudnn-frontend
temp_dir = tempfile.mkdtemp()
cudnn_include_dir = [str(os.path.join(temp_dir, "include"))]


git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("git executable not found")

"""
nvcc_flags = [
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    "--generate-code=arch=compute_80,code=[sm_80]",
]
"""
nvcc_flags = []
ld_flags = ["cuda"]
ext_modules = []


subprocess.run(
    [  # noqa: S603
        git_path,
        "clone",
        "https://github.com/NVIDIA/cudnn-frontend.git",
        "--branch",
        "v1.12.0",
        temp_dir,
    ]
)


ext_modules.append(
    CUDAExtension(
        name="conv_bias",
        sources=["csrc/conv_bias.cpp"],
        include_dirs=cudnn_include_dir,
        extra_compile_args={"nvcc": nvcc_flags},
        libraries=ld_flags,
    )
)

setup(
    name="physicsnemo_custom_ops",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
