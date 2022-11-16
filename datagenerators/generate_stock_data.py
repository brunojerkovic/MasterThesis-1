import numpy as np
import yaml

import yfinance as yf
import pandas as pd

from datagenerators.generator import DataGenerator


class StockGenerator(DataGenerator):
    def __init__(self, config):
        self.config = config
        super(StockGenerator, self).__init__(config)
        self.dataset_path = 'data/stock_data/stock_returns.csv'
        self.stock_list_path = 'configs/stocks.yaml'

        self.__download_stocks()
        print()

    def __download_stocks(self):
        # Download stocks from yfinance
        with open(self.stock_list_path) as f:
            stocks = yaml.load(f, Loader=yaml.FullLoader)['stocks']
        short_names = [stock[0] for stock in stocks]
        data = yf.download(tickers=short_names, start=self.config.stock_start, end=self.config.stock_end, interval=self.config.stock_freq)['Close']

        # Format the data
        data = (np.log(data / data.shift(1)) ** 2).iloc[1:]
        if self.config.stock_n_obs == '1m':
            data['date'] = [str(date.year) + '-' + (str(date.month) if date.month>9 else '0'+str(date.month)) for date in data.index]
        elif self.config.stock_n_obs == '1d':
            data['date'] = [str(date.year) + '-' + (str(date.month) if date.month > 9 else '0' + str(date.month)) + '-' + (str(date.day) if date.day > 9 else '0'+str(date.day)) for
                            date in data.index]
        data = data.groupby(by='date').sum()

        # Save the data
        data.to_csv(self.dataset_path, index=True)

    def _generate_series(self) -> tuple[np.ndarray, np.ndarray]:
        # Load data
        data = pd.read_csv(self.dataset_path)
        series = np.array(data[[col for col in data.columns if col != 'date']])
        n_stocks = series.shape[1]

        # Instantiate coef_mat
        coef_mat = np.ones((n_stocks, n_stocks))

        return series, coef_mat
