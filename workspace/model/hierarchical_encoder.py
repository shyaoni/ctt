import numpy as np
import torch
import torch.nn as nn

from utils import Config, get_instance, dynamic_rnn

class HierarchicalRNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        config = Config(kwargs)
        self.minor_encoder = get_instance(config.minor_encoder)

        if hasattr(config, "major_encoder"):
            self.major_encoder = get_instance(config.major_encoder)
        elif hasattr(config, 'encoder_config_share'):
            self.major_encoder = get_instance(config.minor_encoder)
        else:
            self.major_encoder = None

        self.config = config

    def forward(self, inputs, length_minor, length_major): 
        inputs_to_minor = inputs.view((-1, ) + inputs.shape[2:])
        
        _, minor_codes = dynamic_rnn(
            self.minor_encoder, inputs_to_minor, length_minor.view(-1),
            batch_first=True)

        inputs_to_major = minor_codes.view(
            inputs.shape[:2] + minor_codes.shape[1:])

        if self.major_encoder is None:
            return _, inputs_to_major.squeeze(1)
        
        _, major_codes = dynamic_rnn(
            self.major_encoder, inputs_to_major, length_major,
            batch_first=True)

        return _, major_codes
