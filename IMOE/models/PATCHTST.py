import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, activation='relu'):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)
        self.activation = nn.ReLU()
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = x.unfold(1, self.patch_size, self.patch_size) 
        x = self.projection(x) 
        x = self.activation(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.patch_size = args.patch_size
        self.d_model = args.d_model
        self.num_patches = args.seq_len // args.patch_size
        self.patch_embedding = PatchEmbedding(args.patch_size, args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(args.d_model, 4, args.d_ff, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.fc = nn.Linear(self.num_patches * args.d_model, args.pred_len)

    def forward(self, capacity_increment, relaxation_features, charge_current, discharge_current, Temperature):
        x = self.patch_embedding(capacity_increment)  
        x = self.transformer_encoder(x)  
        x = x.reshape(x.shape[0], -1)  
        x = self.fc(x)  
        return x,x