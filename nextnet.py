import math

import torch
from torch import nn

from models.modules import ConvNextBlock, SinusoidalPosEmb, NormConv1d


class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """
    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, frame_conditioned=False):
        """
        Args:
            in_channels (int):
                Number of input image channels.
            out_channels (int):
                Number of network output channels.
            depth (int):
                Number of ConvNext blocks in the network.
            filters_per_layer (int):
                Base dimension in each ConvNext block.
            frame_conditioned (bool):
                Whether to condition the network on the difference between the current and previous frames. Should
                be True when training a DDPM frame predictor.
        """
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        emb_dim = time_dim * 2 if frame_conditioned else time_dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], emb_dim=emb_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.NormConv1d(dims[depth - 1], 2, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        if frame_conditioned:
            # Encoder for positional embedding of framea
            self.frame_encoder = nn.Sequential(
                 SinusoidalPosEmb(time_dim),
                 nn.Linear(time_dim, time_dim * 4),
                 nn.GELU(),
                 nn.Linear(time_dim * 4, time_dim)  # Adjusting output size to match time_embedding
            )

    def forward(self, x, t, frame_diff=None):
        time_embedding = self.time_encoder(t)

        if frame_diff is not None:
            # Ensure frame_diff has the same batch size as x
            assert frame_diff.shape[0] == x.shape[0], "Batch sizes of frame_diff and input x don't match"
            
            embedding = time_embedding
            """
            frame_embedding = self.frame_encoder(frame_diff)
            time_embedding = time_embedding.unsqueeze(-1)
            frame_embedding = frame_embedding.unsqueeze(-1)
            time_embedding = time_embedding.unsqueeze(-1)
            
            # Concatenate time_embedding and frame_embedding along the feature dimension
            embedding = torch.cat([time_embedding, frame_embedding], dim=1)  # Change dim=2 to dim=1 for concatenation along the correct dimension"""
        else:
            embedding = time_embedding

        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding)

        return self.final_conv(x)
