from typing import Optional, Tuple

import torch
import dgl
import torch.nn as nn
from os import path
import sys
import torch.nn.functional as F
from torch.nn import LayerNorm
from fairseq.modules import FairseqDropout, LayerDropModuleList
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias, PositionalEncoding
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from data.eval_util import read_spectrum
num_units = 512
activation_func = F.relu


class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """

    def __init__(self, vocab_size, ion_size):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(vocab_size * ion_size, num_units, 1)
        self.conv2 = nn.Conv1d(num_units, 2 * num_units, 1)
        self.conv3 = nn.Conv1d(2 * num_units, 4 * num_units, 1)
        self.fc1 = nn.Linear(4 * num_units, 2 * num_units)
        self.fc2 = nn.Linear(2 * num_units, num_units)

        self.output_layer = nn.Linear(num_units, vocab_size)
        self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(vocab_size * ion_size)

        self.bn1 = nn.BatchNorm1d(num_units)
        self.bn2 = nn.BatchNorm1d(2 * num_units)
        self.bn3 = nn.BatchNorm1d(4 * num_units)
        self.bn4 = nn.BatchNorm1d(2 * num_units)
        self.bn5 = nn.BatchNorm1d(num_units)

    def forward(self, x):
        """

        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        x = self.input_batch_norm(x)
        x = activation_func(self.bn1(self.conv1(x)))
        x = activation_func(self.bn2(self.conv2(x)))
        x = activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling

        x = activation_func(self.bn4(self.fc1(x)))
        x = activation_func(self.bn5(self.fc2(x)))

        # x = self.output_layer(x)  # [batch * T, 26]
        return x


class ionCNN(nn.Module):
    """
    encode spectrum info for each candidate
    """

    def __init__(self, encoder_embed_dim, ion_mass, sugar_classes):
        super(ionCNN, self).__init__()
        # self.conv1 = nn.Conv2d(ion_size, out_channel, (3, 5), stride=(1, 1), padding=1)
        # self.activation1 = nn.ReLU()
        self.ion_mass = ion_mass
        self.vocab_size = ion_mass.shape[0]
        self.ion_size = ion_mass.shape[1]
        self.sugar_classes = sugar_classes
        self.encoder_embed_dim = encoder_embed_dim
        self.embed_out = nn.Linear(self.vocab_size*self.ion_size, self.encoder_embed_dim, bias=False)
        self.t_net = TNet(self.vocab_size, self.ion_size)
        self.distance_scale_factor = nn.Parameter(torch.tensor(0).float(), requires_grad=True)

    def forward(self, batched_data, ):
        graphs = dgl.unbatch(batched_data['graph'])
        device = graphs[0].device
        batch_size = len(graphs)
        observed_mz = batched_data["observed_mz"]
        theoretical_mz = batched_data["theoretical_mz"]
        intensity_lists = batched_data["intensity"]
        spec_size = observed_mz.shape[1]
        observed_mz = observed_mz.view(batch_size, spec_size, 1).repeat((1, 1, self.vocab_size*self.ion_size))
        observed_mz = torch.stack([observed_mz] * 3, -1)
        theoretical_mz = theoretical_mz.view(batch_size, 1, self.vocab_size*self.ion_size).repeat(1, spec_size, 1)
        theoretical_left_shift = theoretical_mz - 1
        theoretical_right_shift = theoretical_mz + 1
        theoretical_mz = torch.stack((theoretical_mz, theoretical_left_shift, theoretical_right_shift), -1)

        C_const = 10.0
        delta = torch.abs(observed_mz - theoretical_mz)
        delta_C = -delta * C_const
        sigma = torch.exp(delta_C)
        sigma_flatten = torch.flatten(sigma, start_dim=-2, end_dim=-1)
        ppm_diff = (observed_mz - theoretical_mz)/ (theoretical_mz + 1e-6) * 1e6
        # print(ppm_diff)

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs(
                ppm_diff / 5 * torch.sigmoid(self.distance_scale_factor)
            )
        )
        location_exp_minus_abs_diff, _ = torch.max(location_exp_minus_abs_diff, -1)
        input_feature = torch.cat((location_exp_minus_abs_diff, intensity_lists.unsqueeze(-1)), dim=-1)
        input_feature = location_exp_minus_abs_diff.view(batch_size, spec_size, self.vocab_size*self.ion_size)
        input_feature = input_feature.transpose(1, 2)

        result = self.t_net(input_feature).view(batch_size, self.encoder_embed_dim)
        return result
