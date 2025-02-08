# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# =========================================================================


from re import T
import numpy as np
from torch import nn
from fuxictr.pytorch.torch_utils import get_activation
import torch

class Deepseek_MLP_Block(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[],
                 temp_units=None,
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None, 
                 dropout_rates=0.0,
                 batch_norm=False, 
                 bn_only_once=False, # Set True for inference speed up
                 use_bias=True):
        super(Deepseek_MLP_Block, self).__init__()
        self.dense_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.activations = []
        
        if temp_units is None:
            temp_units = [i * 4 for i in hidden_units]
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units

        if batch_norm and bn_only_once:
            self.bn_layers.append(nn.BatchNorm1d(input_dim))

        for idx in range(len(hidden_units) - 1):
            self.dense_layers.append(nn.Linear(hidden_units[idx], temp_units[idx], bias=use_bias))
            self.gate_layers.append(nn.Linear(hidden_units[idx], temp_units[idx], bias=use_bias))
            self.dense_layers.append(nn.Linear(temp_units[idx], hidden_units[idx + 1], bias=use_bias))
            
            if batch_norm and not bn_only_once:
                self.bn_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            else:
                self.bn_layers.append(None)

            self.activations.append(hidden_activations[idx])
            self.dropout_layers.append(nn.Dropout(p=dropout_rates[idx]) if dropout_rates[idx] > 0 else None)

        if output_dim is not None:
            self.output_layer = nn.Linear(hidden_units[-1], output_dim, bias=use_bias)
        else:
            self.output_layer = None

        if output_activation is not None:
            self.output_activation = get_activation([output_activation])[0]
        else:
            self.output_activation = None

    def forward(self, x):
        """
        定义前向传播过程，使用 gate 对 temp 进行按位乘法。
        """
        for idx in range(len(self.dense_layers) // 2):  # 每个模块有 dense 和 gate 两层
            temp = self.dense_layers[idx * 2](x)  # 计算 temp
            gate = torch.sigmoid(self.gate_layers[idx](x))  # gate 使用 sigmoid 激活
            x = temp * gate  # 按位乘法
            x = self.dense_layers[idx * 2 + 1](x)  # 通过下一个 dense 层

            if self.bn_layers[idx] is not None:
                x = self.bn_layers[idx](x)  # 批归一化

            if self.activations[idx] is not None:
                x = self.activations[idx](x)  # 激活函数

            if self.dropout_layers[idx] is not None:
                x = self.dropout_layers[idx](x)  # Dropout

        if self.output_layer is not None:
            x = self.output_layer(x)  # 输出层

        if self.output_activation is not None:
            x = self.output_activation(x)  # 输出激活函数

        return x