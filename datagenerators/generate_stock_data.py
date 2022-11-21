import numpy as np
import yaml
import os
import yfinance as yf
import pandas as pd

from datagenerators.generator import DataGenerator


class StockGenerator(DataGenerator):
    def __init__(self, config):
        self.config = config
        super(StockGenerator, self).__init__(config)
        self.dataset_dir = 'data/stock_data_complete'
        self.stock_list_path = 'configs/stocks.yaml'

    def _generate_series(self) -> tuple:
        if not self.config.stock_force_preprocessing and os.path.exists('data/stock_data_preprocessed') and len(os.listdir('data/stock_data_preprocessed')) > 0:
            series = np.load('data/stock_data_preprocessed/stocks.npy')
        else:
            # Get only the dates that are present everywhere
            dates_all = [set(pd.read_csv(os.path.join(self.dataset_dir, dataset_path)).Date) for dataset_path in os.listdir(self.dataset_dir)]
            dates = dates_all[0]
            for date in dates_all[1:]:
                dates = dates_all[0] & date
            dates = list(dates)

            # Get all datasets
            datasets = []
            for dataset_path in os.listdir(self.dataset_dir)[:3]:
                data = pd.read_csv(os.path.join(self.dataset_dir, dataset_path))[['Date', 'Time', 'Close']]
                data = data[data['Date'].isin(dates)]
                data[['Date', 'Time']] = data[['Date', 'Time']].astype(str)

                data['Date'] = pd.to_datetime(data.Date, format="%m/%d/%Y")
                data = data[data['Date'].between(self.config.stock_start, self.config.stock_end)]

                # Uzmi one di je 10min razmaka
                data['Time'] = data['Time'].apply(lambda x: x if len(x) == 4 else '0' + x)
                data = data[data['Time'].str.endswith('0')]

                data['Close'] = np.log(data['Close'] / data['Close'].shift(1)) ** 2
                data = data.iloc[1:]
                rv = data.groupby(by='Date').sum()
                datasets.append(np.array(rv).squeeze())

            series = np.array(datasets).T

            # Save preprocessed data
            save_dir_path = os.path.join('data', 'stock_data_preprocessed')
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, 'stocks.npy'), series)
        # Instantiate coef_mat
        coef_mat = np.ones((series.shape[1], series.shape[1]))

        return series, coef_mat
