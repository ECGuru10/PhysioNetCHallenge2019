#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

from torch.nn import LayerNorm
#from .normalize import LayerNorm

"""
Implementation of LSTM variants.

For now, they only support a sequence size of 1, and are ideal for RL use-cases. 
Besides that, they are a stripped-down version of PyTorch's RNN layers. 
(no bidirectional, no num_layers, no batch_first)
"""


class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf

    Special args:
    
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep))).cuda()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]


        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)


        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t =th.mul(h_t, self.mask)
                    h_t *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)




class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, 
                 dropout_method='pytorch', ln_preact=True, learnable=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size, 
                                            hidden_size=hidden_size, 
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)

        
        if ln_preact:
            self.ln_i2h = LayerNorm(4*hidden_size, elementwise_affine=True)
            self.ln_h2h = LayerNorm(4*hidden_size, elementwise_affine=True)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size,elementwise_affine=True)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

       

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)



        c_t = self.ln_cell(c_t)
        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t =th.mul(h_t, self.mask)
                    h_t *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)


class LayerNormGalLSTM(LayerNormLSTM):

    """
    Mixes GalLSTM's Dropout with Layer Normalization
    """
    def __init__(self, *args, **kwargs):
        super(LayerNormGalLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'gal'
        self.sample_mask()


