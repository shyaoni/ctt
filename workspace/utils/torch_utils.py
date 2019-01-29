import numpy as np
import torch
import importlib

_pack_pad_flags = {}

def pack_padded_sequence(input, lengths, batch_first=False, id='temp'): 
    lengths, indices = torch.sort(lengths, descending=True)
    if batch_first:
        input = input[indices]
    else:
        input = input[:, indices]
    _, indices_inv = torch.sort(indices)
    _pack_pad_flags[id] = (indices, indices_inv)

    return torch.nn.utils.rnn.pack_padded_sequence(
        input, lengths, batch_first)

def pad_packed_sequence(sequence,
                        batch_first=False,
                        padding_value=0.0,
                        total_length=None,
                        id='temp'):
    input, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        sequence, batch_first, padding_value, total_length)
    lengths = lengths[pack_order_inv(id)]
    if batch_first:
        input = input[pack_order_inv(id)]
    else:
        input = input[:, pack_order_inv(id)]

    return input, lengths

def pack_order(id='temp', inv=False):
    if inv:
        return _pack_pad_flags[id][1]
    else:
        return _pack_pad_flags[id][0]

def pack_order_inv(id='temp'):
    return pack_order(id, inv=True)

def dynamic_rnn(rnn, inputs, length, batch_first=False):
    outputs, codes = rnn(pack_padded_sequence(
        inputs, length, rnn.batch_first))
    outputs, length = pad_packed_sequence(outputs, rnn.batch_first)
    codes = codes.permute(1, 0, 2)[pack_order_inv()].view(
        len(pack_order_inv()), -1)
    return outputs[pack_order_inv()], codes

############# mask utils ################

def mask_with_length(tensor, length):
    max_length = tensor.shape[len(length.shape)]
    mask = torch.tensor(
        [[1,]*x.item() + [0,]*(max_length-x.item()) for x in length.view(-1)],
        dtype=tensor.dtype,
        device=tensor.device).view(*length.shape, max_length)
    return tensor * mask.view(
        *mask.shape, *([1,]*(len(tensor.shape)-len(mask.shape))))

def mask_with_indexlst(tensor, pts):
    zeros = torch.zeros_like(tensor)

    cond = tensor.zeros(tensor, dtype=torch.uint8)
    for x, lst in enumerate(pts):
        for y in lst:
            cond[x, y] = 1

    return torch.where(cond, tensor, zeros)

def logits_from_label(target, max_length):
    logits = torch.zeros(*target.shape, max_length,
        device=target.device).view(-1)
    base = 0
    for i in target.view(-1).tolist():
        logits[base + i] = 1.
        base += max_length
    return logits.view(*target.shape, max_length)

def sigmoid_cross_entropy_with_logits(input, target, length):
    t = logits_from_label(target, input.shape[-1])
    return mask_with_length(
        torch.max(input, torch.zeros(1, device=input.device)) - \
        input*t + torch.log(1 + torch.exp(-input.abs())), length).sum() / \
            length.sum().item()
