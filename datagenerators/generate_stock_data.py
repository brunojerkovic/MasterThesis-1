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
        if self.config.stock_force_preprocessing or not os.path.exists('data/stock_data_preprocessed') or len(os.listdir('data/stock_data_preprocessed')) == 0:
            dataset_locations = [os.path.join(self.dataset_dir, dataset_path) for dataset_path in os.listdir(self.dataset_dir)] # Get dataset names
            datasets = [pd.read_csv(dataset_location)[['Date', 'Time', 'Close']] for dataset_location in dataset_locations] # Load datasets
            datasets = [df.rename(columns={'Close': dataset_name.split('.')[0]}) for (df, dataset_name) in zip(datasets, os.listdir(self.dataset_dir))] # Rename columns

            # Join datasets
            dataset = datasets[0].set_index(['Date', 'Time'])
            for df_other in datasets[1:]:
                dataset = dataset.join(df_other.set_index(['Date', 'Time']))
            dataset = dataset.dropna()
            dataset = dataset.reset_index()

            # Data preprocessing
            dataset['Date'] = pd.to_datetime(dataset.Date, format="%m/%d/%Y")
            #dataset = dataset[dataset.Date.between(self.config.stock_start, self.config.stock_end)]

            dataset['Time'] = dataset['Time'].astype(str)
            dataset['Time'] = dataset['Time'].apply(lambda x: x if len(x) == 4 else '0' + x)
            dataset = dataset[dataset['Time'].str.endswith('0')]

            closing_prices = dataset.columns.tolist()
            closing_prices.remove('Date')
            closing_prices.remove('Time')
            dataset[closing_prices] = np.log(dataset[closing_prices] / dataset[closing_prices].shift(1)) ** 2
            dataset = dataset.iloc[1:]
            dataset = dataset.groupby(by='Date').sum()

            # Save preprocessed data
            save_dir_path = os.path.join('data', 'stock_data_preprocessed')
            os.makedirs(save_dir_path, exist_ok=True)
            dataset.reset_index().to_csv(os.path.join(save_dir_path, 'stocks.csv'), index=False)

        # Load data
        dataset = pd.read_csv('data/stock_data_preprocessed/stocks.csv')
        dataset['Date'] = pd.to_datetime(dataset.Date, format="%Y-%m-%d")
        dataset = dataset[dataset.Date.between(self.config.stock_start, self.config.stock_end)]
        series = np.array(dataset.loc[:, dataset.columns != 'Date'])
        series = series[:, :self.config.n_stocks]

        # Set config values
        self.config.n_data = series.shape[1]
        self.config.nri['num_atoms'] = self.config.n_data
        self.config.trainset_size = int((series.shape[0] - 2*self.config.timesteps) / (1+2*self.config.tvt_split))

        # Instantiate coef_mat
        coef_mat = np.ones((self.config.n_stocks, self.config.n_data))

        return series, coef_mat
