import torch.nn as nn

# in the process of converting everything to pytorch (way easier)


class Generator(nn.Module):
    def __init__(self, data, latent_size, negative_slope):
        super(Generator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data.shape[1]
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=int(data.shape[1]//1.2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//1.2), out_features=int(data.shape[1]//1.1)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//1.1), out_features=data.shape[1]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, data, negative_slope):
        super(Discriminator, self).__init__()
        self.negative_slope = negative_slope
        self.layers = nn.Sequential(
            nn.Linear(in_features=data.shape[1], out_features=int(data.shape[1]//2)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//2), out_features=int(data.shape[1]//3)),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(in_features=int(data.shape[1]//3), out_features=1),
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




