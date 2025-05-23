#!/usr/bin/env python
# Written by Minh Nguyen and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cbig.Nguyen2020.rnn import MinimalRNNCell
from cbig.Nguyen2020.rnn import LssCell


def jozefowicz_init(forget_gate):
    """
    Initialize the forget gaste bias to 1
    Args:
        forget_gate: forget gate bias term
    References: https://arxiv.org/abs/1602.02410
    """
    forget_gate.data.fill_(1)


class RnnModelInterp(torch.nn.Module):
    """
    Recurrent neural network (RNN) base class
    Missing values (i.e. NaN) are filled using model prediction
    """

    def __init__(self, celltype, nb_classes, nb_measures, h_size, **kwargs):
        super(RnnModelInterp, self).__init__()
        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']

        self.hid2category = nn.Linear(h_size, nb_classes)
        self.hid2measures = nn.Linear(h_size, nb_measures)

        self.cells = nn.ModuleList()
        self.cells.append(celltype(nb_classes + nb_measures, h_size))
        for _ in range(1, kwargs['nb_layers']):
            self.cells.append(celltype(h_size, h_size))

    def init_hidden_state(self, batch_size):
        raise NotImplementedError

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures.out_features, device=dev)
        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    def forward(self, pair_seq):
        _cat_seq, _val_seq = pair_seq[:,:,0:3], pair_seq[:,:,3:]
        out_cat_seq, out_val_seq = [], []

        hidden = self.init_hidden_state(_val_seq.shape[1])
        masks = self.dropout_mask(_val_seq.shape[1])

        cat_seq = _cat_seq.clone().cuda()
        val_seq = _val_seq.clone().cuda()

        for i, j in tqdm(zip(range(len(val_seq)), range(1, len(val_seq)))):
            o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
                                                masks)

            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = torch.isnan(val_seq[j])
            val_seq[j][idx] = (o_val.data[idx]).to(torch.float64)

            idx = torch.isnan(cat_seq[j])
            cat_seq[j][idx] = (o_cat.data[idx]).to(torch.float64)

        return torch.cat((torch.stack(out_cat_seq), torch.stack(out_val_seq)),dim=2)


class SingleStateRNN(RnnModelInterp):
    """
    Base class for RNN model with 1 hidden state (e.g. MinimalRNN)
    (in contrast LSTM has 2 hidden states: c and h)
    """

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state

    def predict(self, i_cat, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = torch.cat([torch.FloatTensor(i_cat.clone().type(torch.FloatTensor)).cuda(), torch.FloatTensor(i_val.clone().type(torch.FloatTensor)).cuda() * i_mask.cuda()],dim=-1)

        next_hid = []
        for cell, prev_h, mask in zip(self.cells, hid, r_mask):
            h_t = cell.cuda()(h_t.cuda(), prev_h.cuda() * mask.cuda())
            next_hid.append(h_t)

        o_cat = nn.functional.softmax(self.hid2category.cuda()(h_t), dim=-1)
        o_val = self.hid2measures.cuda()(h_t) + torch.FloatTensor(i_val.clone().type(torch.FloatTensor)).cuda()

        return o_cat, o_val, next_hid


class user_defined_rnn(nn.Module):
    def __init__(self, nb_classes, nb_measures, h_size, **kwargs):
        super(user_defined_rnn, self).__init__()

        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']

        self.hid2category = nn.Linear(h_size, nb_classes)
        self.hid2measures = nn.Linear(h_size, nb_measures)

        # self.cells = nn.RNN(nb_classes + nb_measures,h_size,kwargs['nb_layers'],batch_first=True)
        # celltype = MinimalRNNCell
        # celltype = nn.RNNCell
        # self.cells = nn.ModuleList()
        # self.cells.append(celltype(nb_classes + nb_measures, h_size))
        # for _ in range(1, kwargs['nb_layers']):
        #     self.cells.append(celltype(h_size, h_size))
        # for cell in self.cells:
        #     jozefowicz_init(cell.bias_hh)
        self.nb_layers = kwargs['nb_layers']
        self.cell_f = nn.LSTMCell(nb_classes + nb_measures, h_size)
    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures.out_features, device=dev)
        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    def forward(self, pair_seq):
        _cat_seq, _val_seq = pair_seq[:,:,0:3], pair_seq[:,:,3:]
        out_cat_seq, out_val_seq = [], []

        hidden = self.init_hidden_state(_val_seq.shape[1])
        masks = self.dropout_mask(_val_seq.shape[1])

        cat_seq = _cat_seq.clone().cuda()
        val_seq = _val_seq.clone().cuda()

        for i, j in tqdm(zip(range(val_seq.shape[0]), range(1, val_seq.shape[0]))):
            o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
                                                masks)

            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = torch.isnan(val_seq[j])
            val_seq[j][idx] = (o_val.data[idx]).to(torch.float64)

            idx = torch.isnan(cat_seq[j])
            cat_seq[j][idx] = (o_cat.data[idx]).to(torch.float64)

        return torch.cat((torch.stack(out_cat_seq), torch.stack(out_val_seq)),dim=2)
    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state

    def predict(self, i_cat, i_val, hid, masks):
        i_mask, r_mask = masks
        i_mask, r_mask = i_mask.cuda(), r_mask
        h_t = torch.cat([torch.FloatTensor(i_cat.clone().type(torch.FloatTensor)).cuda(), torch.FloatTensor(i_val.clone().type(torch.FloatTensor)).cuda() * i_mask],dim=-1)

        next_hid = []
        for cell, prev_h, mask in zip([self.cell_f], hid, r_mask):
            h_t = cell.cuda()(h_t, prev_h.cuda() * mask.cuda())
            next_hid.append(h_t)

        o_cat = nn.functional.softmax(self.hid2category.cuda()(h_t), dim=-1)
        o_val = self.hid2measures.cuda()(h_t) + torch.FloatTensor(i_val.clone().type(torch.FloatTensor)).cuda()

        return o_cat, o_val, next_hid

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
class MinimalRNN(SingleStateRNN):
    """ Minimal RNN """

    def __init__(self, **kwargs):
        super(MinimalRNN, self).__init__(MinimalRNNCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(cell.bias_hh)
class LSS(SingleStateRNN):
    ''' Linear State-Space '''

    def __init__(self, **kwargs):
        super(LSS, self).__init__(LssCell, **kwargs)


class LSTM(RnnModelInterp):
    ''' LSTM '''

    def __init__(self, **kwargs):
        super(LSTM, self).__init__(nn.LSTMCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(
                cell.bias_hh[cell.hidden_size:cell.hidden_size * 2])

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            h_x = torch.zeros(batch_size, cell.hidden_size, device=dev)
            c_x = torch.zeros(batch_size, cell.hidden_size, device=dev)
            state.append((h_x, c_x))
        return state

    def predict(self, i_cat, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = torch.cat([hid[0][0].new(i_cat), hid[0][0].new(i_val) * i_mask],
                        dim=-1)

        states = []
        for cell, prev_state, mask in zip(self.cells, hid, r_mask):
            h_t, c_t = cell(h_t, (prev_state[0] * mask, prev_state[1]))
            states.append((h_t, c_t))

        o_cat = nn.functional.softmax(self.hid2category(h_t), dim=-1)
        o_val = self.hid2measures(h_t) + h_t.new(i_val)

        return o_cat, o_val, states


MODEL_DICT = {'LSTM': LSTM, 'MinRNN': MinimalRNN, 'LSS': LSS}
