import torch
import torch.nn as nn
from models.GeneralBlocks import ConvBlock, NormConv1d


class DiffusionModel(nn.Module):
    def __init__(self, params):
        super(DiffusionModel, self).__init__()
        self.head = ConvBlock(params.hidden_channels, params.hidden_channels, params.filter_size,
                              dilation=params.dilation_factors[0])
        self.body = nn.Sequential()
        for i in range(params.num_layers - 2):
            block = ConvBlock(params.hidden_channels, params.hidden_channels, params.filter_size,
                              dilation=params.dilation_factors[i + 1])
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = NormConv1d(in_channels=params.hidden_channels, out_channels=params.hidden_channels,
                               kernel_size=params.filter_size, dilation=params.dilation_factors[-1])
        self.filter = nn.Sequential(
            NormConv1d(in_channels=params.hidden_channels, out_channels=params.hidden_channels,
                       kernel_size=params.filter_size, padding=int((params.filter_size - 1) / 2)),
            nn.Tanh()
        )
        self.gate = nn.Sequential(
            NormConv1d(in_channels=params.hidden_channels, out_channels=params.hidden_channels,
                       kernel_size=params.filter_size, padding=int((params.filter_size - 1) / 2)),
            nn.Sigmoid()
        )
        self.out_conv = NormConv1d(params.hidden_channels, 1, kernel_size=1)

    def forward(self, x, prev_sig):
        out_head = self.head(x)
        out_body = self.body(out_head)
        out_tail = self.tail(out_body)
        filter = self.filter(out_tail)
        gate = self.gate(out_tail)
        out_tail = filter * gate
        out_tail = self.out_conv(out_tail)
        ind = int((prev_sig.shape[2] - out_tail.shape[2]) / 2)
        prev_sig = prev_sig[:, :, ind:(prev_sig.shape[2] - ind)]
        output = out_tail + prev_sig
        return output
