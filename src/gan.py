import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope):
        super(Generator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=int(data_size//1.2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data_size//1.2), out_features=int(data_size//1.1)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data_size//1.1), out_features=data_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, data_size, negative_slope):
        super(Discriminator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.layers = nn.Sequential(
            nn.Linear(in_features=data_size, out_features=int(data_size//2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data_size//2), out_features=int(data_size//3)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data_size//3), out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        x = self.generator(x)
        return self.discriminator(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_size+self.num_classes, out_features=int(self.data_size//1.2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(self.data_size//1.2), out_features=int(self.data_size//1.1)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(self.data_size//1.1), out_features=self.data_size),
            nn.Tanh()
        )

    def forward(self, x, labels):
        labels = torch.tensor(np.eye(self.num_classes)[labels.numpy().reshape(-1)], dtype=torch.float)
        input_ = torch.cat((x, labels), 1)
        return self.layers(input_)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, data_size, negative_slope, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.data_size+self.num_classes, out_features=int(self.data_size//2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(self.data_size//2), out_features=int(self.data_size//3)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(self.data_size//3), out_features=1),
            nn.Sigmoid()
        )

    # pass example and one-hot encoded label
    def forward(self, x, labels):
        labels = torch.tensor(np.eye(self.num_classes)[labels.numpy().reshape(-1)], dtype=torch.float)
        input_ = torch.cat((x, labels), 1)
        return self.layers(input_)




