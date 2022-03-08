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

import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from Code.Modeling.model import Discriminator
from Code.Modeling.model import Generator

import Code.Data_Acquisition_and_Understanding.dataPrep as dataPrep


# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def efficient_zero_grad(model):
  """
    Apply zero_grad more efficiently
    Source: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  """
  for param in model.parameters():
    param.grad = None


# Calculating gradient penalty for WGAN-GP
def calc_gradient_penalty(netD, real, fake):
    LAMBDA = 0.1

    alpha = torch.rand((real.size(0), 1, 1), device=device)
    alpha = alpha.expand(real.size())

    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates.to(device)

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    interpolates.to(device)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# Training loop
def train_model(netG, netD, train_data_loader, EPOCHS):
    for epoch in range(EPOCHS):
        start = time.time()
        print(f'_______________________\n\nEpoch {epoch + 1}:')

        # used to store all the losses of an epoch of the generator and the discriminator
        train_losses_g = []
        train_losses_d = []

        for batch in tqdm(train_data_loader):
            # Prepare data, labels, noise
            real = batch.to(device)
            batch_size = real.size(0)
            label_real = torch.full((batch_size,), 1.0, device=device)
            label_fake = torch.full((batch_size,), 0.0, device=device)
            noise = torch.randn(batch_size, 100, device=device)

            # Update Discriminator
            # netD.zero_grad()
            efficient_zero_grad(netD)
            output = netD(real).view(-1)
            if type in ['WGAN', 'WGANGP']:
                loss_d_real = -criterion(output)
            else:
                loss_d_real = criterion(output, label_real)
            loss_d_real.backward()

            fake = netG(noise)
            output = netD(fake.detach()).view(-1)
            if type in ['WGAN', 'WGANGP']:
                loss_d_fake = criterion(output)
            else:
                loss_d_fake = criterion(output, label_fake)
            loss_d_fake.backward()

            if type == 'WGANGP':
                gradient_penalty = calc_gradient_penalty(netD, real, fake)
                gradient_penalty.backward()
                loss_d = loss_d_real + loss_d_fake + gradient_penalty
            else:
                loss_d = loss_d_real + loss_d_fake

            optimiserD.step()

            # Update Generator
            # netG.zero_grad()
            efficient_zero_grad(netG)
            output = netD(fake).view(-1)
            if type in ['WGAN', 'WGANGP']:
                loss_g = -criterion(output)
            else:
                loss_g = criterion(output, label_real)
            loss_g.backward()
            optimiserG.step()

            train_losses_g.append(loss_g.item())
            train_losses_d.append(loss_d.item())

        epoch_loss_g = np.mean(train_losses_g)
        epoch_loss_d = np.mean(train_losses_d)
        print(f'Train loss Generator = {epoch_loss_g:.4}')
        print(f'Train loss Diskriminator = {epoch_loss_d:.4}')

        train_epoch_lossesG.append(epoch_loss_g)
        train_epoch_lossesD.append(epoch_loss_d)
        sns.lineplot(data={'Gen': train_epoch_lossesG, 'Dis': train_epoch_lossesD})
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('generator and discriminator loss over time')
        plt.show()
        print(f'Time for Epoch {epoch + 1} is {(time.time() - start):.4} sec')

##############################################################

# Actual training of a model


train_data_loader = dataPrep.train_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Selecting type of model (DCGAN, WGAN, WGANGP)
type = 'DCGAN'

# Initialize models
discriminator = Discriminator()
generator = Generator()

netD = discriminator.to(device)
netG = generator.to(device)

# Select type of loss according to type of model
if type in ['WGAN', 'WGANGP']:
    criterion = torch.mean
else:
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion.to(device)

# Initialize weights
netG.apply(weights_init)
netD.apply(weights_init)

# Set models to training state
netG.train()
netD.train()

# Select optimizer according to type of model
if type == 'WGAN':
    optimiserD = torch.optim.RMSprop(netD.parameters(), lr=0.00005)
    optimiserG = torch.optim.RMSprop(netG.parameters(), lr=0.00005)
elif type == 'WGANGP':
    optimiserD = torch.optim.Adam(netD.parameters(), lr=0.0005, betas=(0., 0.9))
    optimiserG = torch.optim.Adam(netG.parameters(), lr=0.0005, betas=(0., 0.9))
else:
    optimiserD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimiserG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Globale variables to keep track of the losses per epoch (accessed in train_model)
train_epoch_lossesG = []
train_epoch_lossesD = []

EPOCHS = 400

train_model(netG, netD, train_data_loader, EPOCHS)

# Save model as .pt and model parameters as .txt
datetime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = f'Trained_Models/generator_{datetime}_{type}_{EPOCHS}_{dataPrep.name}'
torch.save(netG, f'{filename}.pt')

f = open(f'{filename}.txt', 'w+')
f.write(f'Type: {type}\n')
f.write(f'Epochs: {EPOCHS}\n')
f.write(f'Batch Size: {dataPrep.batch_size}\n\n')
f.write(f'Index Name: {dataPrep.name}\n')
f.write(f'Index Start Date: {dataPrep.start_date}\n')
f.write(f'Index End Date: {dataPrep.end_date}\n\n')
f.write(f'Datetime: {datetime}\n')
f.close()

