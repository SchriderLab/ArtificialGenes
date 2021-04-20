from torch.utils.data import Dataset


class GenomesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples
