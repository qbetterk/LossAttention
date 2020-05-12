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
                                                   nhead=2,
                                                   dim_feedforward=self.hidden_size,
                                                   dropout=cfg.dropout)
        self.trans = nn.TransformerEncoder(decoder_layer, num_layers=1)

        self.linear = nn.Linear(input_size, 1)


    def forward(self, true_enc, usdx_enc):
        """
        input: ground truth, context, utt
        output: weight
        """
        trans_input = torch.cat([true_enc, usdx_enc], 1)

        trans_output = self.trans(trans_input)

        weights = F.softmax(self.linear(trans_output[:,:true_enc.shape[1],:]), dim=1) * 10 + 0.5

        return weights