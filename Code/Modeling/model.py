#   GAN4FIN -  Generative Adversarial Network for Finance
#   Copyright (C) 2021  Timo Kühne, Jonathan Laib
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch.nn as nn
import torch.utils.data

from torchsummary import summary

BUFFER_SIZE = 5032  # lengths of timeseries
BATCH_SIZE = 32
data_dim = 32
noise_dim = 100
data_channel = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = 1
        self.gkernel = 5
        self.gf_dim = 32

        self.layers_g = nn.Sequential(
            nn.Linear(noise_dim, data_dim * self.gf_dim),
            nn.Unflatten(1, (256, 4)),

            # Block 1
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=self.gkernel, stride=2, padding=2,
                               output_padding=1),

            # Block 2
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=self.gkernel, stride=2, padding=2,
                               output_padding=1),

            # Block 3
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=self.gkernel, stride=2, padding=2,
                               output_padding=1),

            # Block 4
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=self.gkernel, stride=1, padding=2)
        )

    def forward(self, input):
        return self.layers_g(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = 1
        self.dkernel = 5
        self.df_dim = 32

        self.layers_d = nn.Sequential(
            nn.Conv1d(1, self.df_dim, kernel_size=self.dkernel, stride=2, padding=2),
            nn.LeakyReLU(),

            # Block 1
            nn.Conv1d(32, 64, kernel_size=self.dkernel, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=self.dkernel, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(512, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers_d(input)


# Summery of models and their layers and parameters 
def print_summary():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator()
    generator.to(device)
    print(summary(generator.to(device), (100,)))

    noise = torch.randn(1, 100, device=device)

    fake = generator(noise)

    discriminator = Discriminator()
    discriminator.to(device)
    print(summary(discriminator.to(device), (1, 32)))
