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

import numpy as np
import pandas as pd
import pandas_datareader as data_reader
import torch

ticker = '^GDAXI'
name = 'DAX'
start_date = '2001-01-01'
end_date = '2022-01-01'
ohlc_data = 'Adj Close'
window_size = 32
stride = 1
batch_size = 32

# Downloading Data from Yahoo Finance
prices_returns = pd.DataFrame()
prices_returns[name] = data_reader.DataReader(ticker, 'yahoo', start_date, end_date)[ohlc_data]
prices_returns[name] = prices_returns[name].fillna(method='ffill')
prices_returns['log_return'] = np.log(prices_returns[name] / prices_returns[name].shift(1)) * 100
prices_returns['log_return'] = prices_returns['log_return'].fillna(0)
prices_returns['return'] = prices_returns[name] / prices_returns[name].shift(1) * 100

# Loading Data into DataLoader
train_data = torch.tensor(prices_returns['log_return']).unfold(0, window_size, stride)
train_data = torch.reshape(train_data, (train_data.size(0), 1, window_size))
train_data = train_data.type(torch.FloatTensor)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
