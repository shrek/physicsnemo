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

from typing import Any, Callable, Dict, List

import torch
from torch._inductor import config

from physicsnemo.compile.conv_bias_op import conv_bias_fprop
import physicsnemo.compile.conv_bias_op as conv_bias_op  # noqa: F401


class PhysicsNemoBackend:
    """
    A custom PyTorch backend for PhysicsNemo that provides specialized compilation
    and optimization passes for physics-based neural networks.

    This backend extends PyTorch's compilation capabilities with custom fusion
    passes, particularly for convolution operations with bias terms, to improve
    performance on physics simulation workloads.

    Attributes:
        cfg (Dict[str, Any]): Configuration dictionary containing backend settings
            such as enable_conv_bias_fusion and amp_mode flags.
    """


    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the PhysicsNemoBackend with configuration settings.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary containing backend options.
                Expected keys include:
                - enable_conv_bias_fusion (bool): Whether to enable conv-bias fusion
                - amp_mode (bool): Whether automatic mixed precision is enabled
        """
        self.cfg = cfg

    def print_node(self, node: torch.fx.Node):
        """
        Print detailed information about a PyTorch FX node for debugging purposes.

        Args:
            node (torch.fx.Node): The FX node to print information about.
        """
        print(f"node: {node}")
        print(f"node.target: {node.target}")
        print(f"node.args: {node.args}")
        print(f"node.kwargs: {node.kwargs}")
        print(f"node.name: {node.name}")
        print(f"node.op: {node.op}")

    def conv_bias_fusion_pass(self, graph: torch.fx.Graph) -> torch.fx.Graph:
        """
        Apply convolution-bias fusion optimization pass to the graph.

        This pass identifies torch.conv2d operations with non-None bias parameters
        and replaces them with fused conv_bias_fprop operations for improved
        performance. The fusion is only applied when the convolution has standard
        parameters (no groups, has bias and padding).

        Args:
            graph (torch.fx.Graph): The FX graph to apply the fusion pass to.

        Returns:
            torch.fx.Graph: The modified graph with fused convolution operations.
        """
        print("total graph nodes: ", len(graph.nodes))
        conv2dnodes = [n for n in graph.nodes if n.target == torch.conv2d]
        # print(f"conv2dnodes: {len(conv2dnodes)}")
        replaced_nodes = []
        for node in conv2dnodes:
            if (
                "groups" not in node.kwargs
                and "bias" in node.kwargs
                and "padding" in node.kwargs
            ):
                bias = node.kwargs["bias"]
                padding = node.kwargs["padding"]
                stride = 1
                args = node.args + (bias, padding, stride)
                # replace the node with the fused node
                with graph.inserting_after(node):
                    fused_node = graph.create_node(
                        "call_function",
                        torch.ops.physicsnemo.conv_bias_fprop,
                        args=args,
                        name=f"{node.name}_fused",
                    )
                    node.replace_all_uses_with(fused_node)
                replaced_nodes.append(node)
        for node in replaced_nodes:
            graph.erase_node(node)
        print("replaced nodes: ", len(replaced_nodes))
        return graph

    def replace_conv_bias_pattern(self, gm: torch.fx.GraphModule):
        """
        Recursively apply conv-bias fusion to all modules in the graph module.

        This method traverses the entire graph module hierarchy and applies
        the conv_bias_fusion_pass to each submodule that has a graph attribute.

        Args:
            gm (torch.fx.GraphModule): The graph module to process recursively.
        """
        gm.graph = self.conv_bias_fusion_pass(gm.graph)
        for submodule_name, submodule in gm.named_children():
            if hasattr(submodule, "graph"):
                # print("replacing in child module: ", submodule_name)
                self.replace_conv_bias_pattern(submodule)

    def backend(self) -> Callable:
        """
        Create and return the backend compilation function.

        This method returns a callable that can be used as a PyTorch backend
        for torch.compile(). The returned function applies custom optimization
        passes (like conv-bias fusion) before delegating to PyTorch's inductor
        compiler.

        Returns:
            Callable: A function that takes a GraphModule and inputs, applies
                custom optimizations, and returns a compiled callable.
        """

        def fn(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]) -> Callable:
            """
            Compile a PyTorch GraphModule with custom optimizations.

            This function applies PhysicsNemo-specific optimization passes (such as
            conv-bias fusion) to the graph module before delegating compilation to
            PyTorch's inductor compiler. The function handles configuration management
            and conditional optimization based on backend settings.

            Args:
                gm (torch.fx.GraphModule): The graph module to compile. This contains
                    the computational graph representation of the model.
                inputs (List[torch.Tensor]): List of input tensors that will be used
                    during compilation to determine shapes and optimize the graph.

            Returns:
                Callable: A compiled callable that can be executed with the same
                    input signature as the original model, but with optimizations
                    applied for improved performance.
            """
            current_config = config.shallow_copy_dict()
            from torch._inductor.compile_fx import compile_fx

            if self.cfg.get("enable_conv_bias_fusion", False):
                self.replace_conv_bias_pattern(gm)

            # gm.print_readable()
            # graph.print_readable()
            return compile_fx(gm, inputs, config_patches=current_config)

        return fn
