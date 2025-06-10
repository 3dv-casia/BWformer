import torch
import math
import numpy as np


def get_emb(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def positional_encoding_3d(d_model, height, width, length):
    if d_model % 6 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    d_model = int(np.ceil(d_model / 6) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    pos_x = torch.arange(height).type(inv_freq.type())
    pos_y = torch.arange(width).type(inv_freq.type())
    pos_z = torch.arange(length).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
    emb_y = get_emb(sin_inp_y).unsqueeze(1)
    emb_z = get_emb(sin_inp_z)
    emb = torch.zeros((height, width, length, d_model * 3))
    emb[:, :, :, : d_model] = emb_x
    emb[:, :, :, d_model : 2 * d_model] = emb_y
    emb[:, :, :, 2 * d_model :] = emb_z
    emb = emb.permute(3,0,1,2)
    return emb

def positional_encoding_2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def positional_encoding_1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe





