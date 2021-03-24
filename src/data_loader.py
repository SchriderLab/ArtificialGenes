### still writing and debugging this script ###

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math


class GenomesDataset(Dataset):
    def __init__(self, data_path):
        # don't need to worry about labels yet
        if ".hapt" in data_path:
            data = pd.read_csv(data_path, sep=' ', header=None)
            data = data.reset_index(drop=True)
            data = data.drop(data.columns[0:2], axis=1).values
        else:
            data = pd.read_csv(data_path)
            # test just sequence from rows 7k-8k
            # data = data.iloc[7000:7805, :].T
            data = data.reset_index(drop=True).values
        self.data = torch.FloatTensor(data - np.random.uniform(0, 0.1, size=(data.shape[0], data.shape[1])))

        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples
