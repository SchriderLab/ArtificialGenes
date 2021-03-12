import torch
import torch.nn as nn


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
    def __init__(self, data, latent_size, negative_slope, n_classes):
        super(ConditionalGenerator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data.shape[1]
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_size+n_classes, out_features=int(data.shape[1]//1.2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//1.2), out_features=int(data.shape[1]//1.1)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//1.1), out_features=data.shape[1]),
            nn.Tanh()
        )

    def forward(self, x, label):
        input_ = torch.cat((x, label), -1)
        return self.layers(input_)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, data, negative_slope, n_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.negative_slope = negative_slope
        self.layers = nn.Sequential(
            nn.Linear(in_features=data.shape[1]+n_classes, out_features=int(data.shape[1]//2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//2), out_features=int(data.shape[1]//3)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//3), out_features=1),
            nn.Sigmoid()
        )

    # pass example and one-hot encoded label
    def forward(self, x, label):
        input_ = torch.cat((x, label), -1)
        return self.layers(input_)




