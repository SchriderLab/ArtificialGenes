from torch.utils.data import Dataset


# for GAN and WGAN
class GenomesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples


# for conditional GAN
class PopulationsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.n_samples = data.shape[0]
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples
