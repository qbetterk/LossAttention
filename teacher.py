import os, sys
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F

import utils
from config import global_config as cfg

import pdb




class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.embedding = embedding
        # self.embed_size = embedding.embedding_dim
        self.hidden_size = cfg.hidden_size

        input_size = self.hidden_size

        decoder_layer = nn.TransformerEncoderLayer(d_model=input_size, 
                                                   nhead=cfg.t_head_num,
                                                   dim_feedforward=self.hidden_size,
                                                   dropout=cfg.dropout)
        self.trans = nn.TransformerEncoder(decoder_layer, num_layers=cfg.t_layer_num)

        self.linear = nn.Linear(input_size, 1)


    def forward(self, true_enc, usdx_enc, resp):
        """
        input: ground truth, context, utt
        output: weight
        """
        trans_input = torch.cat([true_enc, usdx_enc], 1)

        trans_output = self.trans(trans_input)

        linear_out = self.linear(trans_output.narrow(1,0,true_enc.shape[1])).squeeze(2)

        # weights_norm = F.softmax(linear_out, dim=1) * 10 + 0.5

        weights = linear_out.masked_fill(resp==0, float('-inf'))

        weights_norm = F.softmax(weights, dim=1) * 10 + 0.5

        weights_norm2 = weights_norm.masked_fill(resp==0, 0)
        # pdb.set_trace()
        return weights_norm2