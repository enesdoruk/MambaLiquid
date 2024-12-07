from torch.utils.data import Dataset
import pandas as pd
import torch
import os


class StockPrice(Dataset):
    def __init__(self, root, data_file, n_test):
        self.root = root
        self.data_file = data_file
        self.n_test = n_test
        
        self.data = pd.read_csv(os.path.join(self.root, self.data_file) +'.SH.csv')
        self.data['trade_date'] = pd.to_datetime(self.data['trade_date'], format='%Y%m%d')
        self.close = self.data.pop('close').values
        self.ratechg = self.data['pct_chg'].apply(lambda x:0.01*x).values
        self.data.drop(columns=['pre_close','change','pct_chg'],inplace=True)
        self.dat = self.data.iloc[:,2:].values

    def get_data(self):
        x_train, x_test = self.dat[:-self.n_test, :], self.dat[-self.n_test:, :]
        y_train = self.ratechg[:-self.n_test]
        
        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_train = torch.from_numpy(y_train).float()
                
        return x_train, y_train, x_test, self.close, self.data
        