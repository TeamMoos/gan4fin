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

import seaborn as sns
import matplotlib.pyplot as plt

import Code.Data_Acquisition_and_Understanding.dataPrep as dataPrep

figsize = (6, 4)
dpi = 200


def print_analysis(prices_returns):
    # Outputting a few metrics for DAX
    print(prices_returns.describe())

    # Plotting DAX
    fig = plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=prices_returns['DAX'])
    plt.show()
    fig.savefig('images/Prices_Real.png', dpi=fig.dpi)

    # Outputting a few metrics for DAX log returns
    print(prices_returns['log_return'].describe())

    # Plotting DAX log returns
    fig = plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=prices_returns['log_return'])
    plt.show()
    fig.savefig('images/Prices_log_returns_Real.png', dpi=fig.dpi)

    # Plotting Histogram of DAX log returns
    fig = plt.figure(figsize=figsize, dpi=dpi)
    sns.histplot(data=prices_returns['log_return'], bins=200, alpha=0.5, stat="probability")
    plt.title('Histogram Prices log return Real')
    plt.show()
    fig.savefig('images/Histogram_prices_log_returns_Real.png', dpi=fig.dpi)

    # Plotting Histogram of DAX returns
    fig = plt.figure(figsize=figsize, dpi=dpi)
    sns.histplot(data=prices_returns['return'], bins=200, alpha=0.5, stat="probability")
    plt.title('Histogram Prices return Real')
    plt.show()
    fig.savefig('images/Histogram_prices_returns_Real.png', dpi=fig.dpi)


print_analysis(dataPrep.prices_returns)

