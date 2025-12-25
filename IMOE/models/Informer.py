import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from layers.attn import FullAttention, ProbAttention, AttentionLayer
from layers.encoder import Encoder, EncoderLayer, ConvLayer
from layers.embed import DataEmbedding
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
import random
import openpyxl
import os
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.attn = 'prob'
        self.output_attention = False
        self.enc_embedding = DataEmbedding(1, args.d_model, 'fixed', freq='h', dropout=args.dropout)
        distil= True
        Attn = ProbAttention if self.attn == 'prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor=5, attention_dropout=args.dropout, output_attention=False),
                                   args.d_model, n_heads=2, mix=False),
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    dropout=args.dropout,
                    activation='gelu'
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(args.d_model) for l in range(args.e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.fc = nn.Linear(1600, args.pred_len)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, capacity_increment, relaxation_features, charge_current, discharge_current,TEMPERTURE):
        x_enc = capacity_increment.float()
        x_enc = x_enc.squeeze(1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.view(enc_out.shape[0], -1)
        enc_out = self.fc(enc_out)
        return enc_out,enc_out