import torch
import torch.nn as nn

# https://github.com/rajatguptakgp/image_enhancement_techniques


class Denoiser(nn.Module):
    # No pooling because we want to keep the features
    def __init__(self, image_size: int = 16, RGB: bool = False):
        super().__init__()
        if RGB:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, image_size, 3, padding=1),
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
                nn.Conv2d(image_size, 3, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, image_size, 3, padding=1),
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
                nn.Conv2d(image_size, 1, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        return self.decoder(x)


class FullDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
