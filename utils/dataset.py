from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


class StockPrice(Dataset):
    def __init__(self, root, ratio, mode):
        files = os.listdir(root)

        scaler = MinMaxScaler(feature_range=(0,1))

        self.data, self.label = [], []
        for filename in files:
            try:
                df = pd.read_csv(os.path.join(root,filename), sep=',')
                df = df.drop(['Date', 'OpenInt'], axis=1)
                df = df.dropna()
                df_label = df['Close']
                df_data = df.drop(['Close'],axis=1)
                self.data.append(np.array(df_data))
                self.label.append(np.array(df_label))
            except:
                continue
        
        self.data = np.concatenate(self.data, axis=0)
        self.label = np.concatenate(self.label, axis=0)

        if mode == 'train':
            length = self.__len__()
            self.data = self.data[:int(length * ratio)]
            self.label = self.label[:int(length * ratio)]
        if mode == 'test':
            length = self.__len__()
            self.data = self.data[int(length * (1-ratio)):]
            self.label = self.label[int(length * (1-ratio)):]
        
        self.label = scaler.fit_transform(np.array(self.label).reshape(-1,1))
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)