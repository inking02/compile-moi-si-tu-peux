import torch
import torch.nn as nn


class Denoiser(nn.Module):
    # No pooling because we want to keep the features
    def __init__(self, image_size: int = 16, RGB: bool = False, num_layers: int = 4):
        super().__init__()
        if RGB:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
