### still writing and debugging this script ###

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math


class GenomesDataset(Dataset):
    def __init__(self, data):
        # don't need to worry about labels yet
        self.data = data
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples
