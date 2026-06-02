"""
File: classical_denoiser.py

Description: This module contains classical convolutional autoencoders for image
denoising, including a configurable grayscale/RGB autoencoder and a deeper RGB
encoder-decoder model inspired by image enhancement architectures.
"""

import torch
import torch.nn as nn


class Denoiser(nn.Module):
    """
    Small configurable autoencoder for grayscale or RGB denoising.

    Args:
        image_size (int): Number of base convolution channels used by the encoder.
        RGB (bool): If true, build a 3-channel model; otherwise build a
            1-channel model.
    """

    def __init__(self, image_size: int = 16, RGB: bool = False) -> None:
        super().__init__()

        input_channels = 3 if RGB else 1
        output_channels = 3 if RGB else 1

        # Strided convolutions downsample while preserving local image structure.
        if RGB:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, image_size, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(image_size, image_size * 2, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(image_size * 2, image_size * 4, 3, stride=2, padding=1),
                nn.ReLU(),
            )

            # Transposed convolutions restore the original spatial resolution.
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    image_size * 2, image_size, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(image_size, output_channels, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, image_size, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(image_size, image_size * 2, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(image_size * 2, image_size * 4, 3, stride=2, padding=1),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    image_size * 2, image_size, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(image_size, output_channels, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoises a batch of images.

        Args:
            x (torch.Tensor): Image batch shaped ``(batch, channels, height, width)``.

        Returns:
            torch.Tensor: Reconstructed image batch with values constrained to
            ``[0, 1]``.
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)


class Encoder(nn.Module):
    """Convolutional feature extractor for 32x32 RGB images."""

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 64 x 16 x 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 128 x 8 x 8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes RGB images into a high-channel latent representation.

        Args:
            x (torch.Tensor): RGB image batch shaped ``(batch, 3, 32, 32)``.

        Returns:
            torch.Tensor: Encoded feature maps shaped ``(batch, 256, 8, 8)``.
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder that reconstructs 32x32 RGB images from encoder features."""

    def __init__(self) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            # 256 x 8 x 8
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent feature maps back into normalized RGB images.

        Args:
            x (torch.Tensor): Encoded feature maps shaped ``(batch, 256, 8, 8)``.

        Returns:
            torch.Tensor: RGB image batch shaped ``(batch, 3, 32, 32)``.
        """
        return self.decoder(x)


class FullDenoiser(nn.Module):
    """Full RGB denoising autoencoder composed of ``Encoder`` and ``Decoder``."""

    def __init__(self) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns denoised RGB images for an input image batch.

        Args:
            x (torch.Tensor): Noisy RGB image batch shaped ``(batch, 3, 32, 32)``.

        Returns:
            torch.Tensor: Denoised RGB image batch shaped ``(batch, 3, 32, 32)``.
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)
