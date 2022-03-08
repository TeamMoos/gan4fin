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

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import kurtosis
from scipy.stats import skew

import Code.Data_Acquisition_and_Understanding.dataPrep as dataPrep

device = torch.device("cpu")

list_of_files = glob.glob('Trained_Models/*.pt')  # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
latest_file = 'Trained_Models/generator_2022_01_21_15_06_02_DCGAN_250_DAX.pt'

generator = torch.load(latest_file).to(device)
generator.eval()

noise = torch.randn(round(len(dataPrep.prices_returns.index) / dataPrep.batch_size + 0.49), 100, device=device)
prediction = torch.reshape(generator(noise).squeeze(), (-1,))
prediction = prediction[:len(dataPrep.prices_returns.index)]

generated_returns = pd.DataFrame(prediction.detach(), columns=['log_return'], index=dataPrep.prices_returns.index)

manipulated_generated_returns = generated_returns.div(100)
manipulated_generated_returns.iloc[0, 0] = 0
price0 = dataPrep.prices_returns.iloc[0, 0]  # arbitrary value  1335 on 01.01.2001
manipulated_generated_returns["logr_na0"] = manipulated_generated_returns.log_return.fillna(0)
manipulated_generated_returns['cumlog'] = np.cumsum(manipulated_generated_returns.logr_na0)
manipulated_generated_returns["norm"] = np.exp(manipulated_generated_returns.cumlog)
manipulated_generated_returns["prices_back"] = price0 * manipulated_generated_returns.norm

list_generated = manipulated_generated_returns.iloc[:, 4]
list_original = dataPrep.prices_returns.iloc[:, 0]

figsize = (6,4)
dpi = 200

fig = plt.figure(figsize=figsize, dpi=dpi)
sns.lineplot(data={'real': list_original, 'synthetic': list_generated})
plt.title('prices real vs synthetic')
plt.show()
fig.savefig('images/prices_real_vs_synthetic.png', dpi=fig.dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
sns.lineplot(data={'synthetic': generated_returns.log_return})
plt.ylabel('log return')
plt.title('synthetic log return')
plt.show()
fig.savefig('images/synthetic_log_return.png', dpi=fig.dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
sns.lineplot(data={'real': dataPrep.prices_returns.log_return})
plt.ylabel('log return')
plt.title('real log return')
plt.show()
fig.savefig('images/real_log_return.png', dpi=fig.dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
pd.plotting.autocorrelation_plot(dataPrep.prices_returns['log_return'], label='real').set_ylim([-0.1, 0.1])
plt.title('autocorrelation of real log return')
plt.show()
fig.savefig('images/autocorrelation_real_log_return.png', dpi=fig.dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
pd.plotting.autocorrelation_plot(generated_returns, label='synthetic').set_ylim([-0.1, 0.1])
plt.title('autocorrelation synthetic log return')
plt.show()
fig.savefig('images/autocorrelation_synthetic_log_return.png', dpi=fig.dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
sns.histplot(data={'real': dataPrep.prices_returns['log_return'],
                   'synthetic': generated_returns['log_return']}, bins=200, alpha=0.5, stat="probability")
plt.title('histogram real vs synthetic log return')
plt.show()
fig.savefig('images/histogram_real_vs_synthetic_log_return.png', dpi=fig.dpi)

f = open(f'images/skewness_kurtosis.txt', 'w+')
f.write(f'Skewness of real data: {skew(dataPrep.prices_returns.log_return):.4}/n')
f.write(f'Skewness of synthetic data: {skew(generated_returns.log_return):.4}/n')

f.write(f'Kurtosis of real data: {kurtosis(dataPrep.prices_returns.log_return):.4}/n')
f.write(f'Kurtosis of synthetic data: {kurtosis(generated_returns.log_return):.4}/n')
f.close()

print(f'Skewness of real data: {skew(dataPrep.prices_returns.log_return):.4}')
print(f'Skewness of synthetic data: {skew(generated_returns.log_return):.4}')

print(f'Kurtosis of real data: {kurtosis(dataPrep.prices_returns.log_return):.4}')
print(f'Kurtosis of synthetic data: {kurtosis(generated_returns.log_return):.4}')


