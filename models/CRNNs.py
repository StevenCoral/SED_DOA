import math
import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer, Transformer

from models.model_utilities import ConvBlock, init_gru, init_layer, interpolate


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class CRNN10(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(2, 2), interp_ratio=16, pretrained_path=None,
                 dropout: float = 0.1, nhead: int = 8, nlayers: int = 2, kind='Transformer'):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = interp_ratio
        self.kind = kind
        
        self.conv_block1 = ConvBlock(in_channels=10, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.d_model = 512
        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout

        if self.kind == 'Transformer':
            self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
            self.decoder = nn.Linear(self.d_model, 512)
            self.change_encoder(nn.GELU(), self.nhead)

        elif self.kind == 'LSTM':
            self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
            self.lstm = nn.LSTM(input_size=512,
                                hidden_size=256,
                                num_layers=2,
                                dropout=0.1, batch_first=True,
                                bidirectional=True)

        elif self.kind == 'NONE':
            pass

        else:
            self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(512, class_num, bias=True)
        self.azimuth_fc = nn.Linear(512, class_num, bias=True)
        self.elevation_fc = nn.Linear(512, class_num, bias=True)

        self.init_weights()

    def init_weights(self, lstm_layers=2):
        if self.kind == 'Transformer':
            initrange = 0.1
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        elif self.kind == 'LSTM':
            lstm_drop = 0.1 if lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(input_size=512,
                                hidden_size=256,
                                num_layers=1,
                                dropout=lstm_drop, batch_first=True,
                                bidirectional=True)

        elif self.kind == 'NONE':
            pass

        else:
            init_gru(self.gru)

        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1, 2)
        ''' (batch_size, time_steps, feature_maps):'''

        if self.kind == 'Transformer':
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = self.decoder(x)

        elif self.kind == 'LSTM':
            x = self.pos_encoder(x)
            x, _ = self.lstm(x)

        elif self.kind == 'NONE':
            pass

        else:
            (x, _) = self.gru(x)

        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output

    def change_encoder(self, activation, nhead):
        encoder_layers = TransformerEncoderLayer(self.d_model, 2**nhead, 256, self.dropout, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)



class pretrained_CRNN10(CRNN10):

    def __init__(self, class_num, pool_type='avg', pool_size=(2, 2), interp_ratio=16, pretrained_path=None):

        super().__init__(class_num, pool_type, pool_size, interp_ratio, pretrained_path)
        
        self.load_weights(pretrained_path)

    def load_weights(self, pretrained_path):

        model = CRNN10(self.class_num, self.pool_type, self.pool_size, self.interp_ratio)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4

        self.init_weights()

        # init_gru(self.gru)
        # init_layer(self.event_fc)
        # init_layer(self.azimuth_fc)
        # init_layer(self.elevation_fc)
