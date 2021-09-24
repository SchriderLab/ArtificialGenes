import torch
import torch.nn as nn
import numpy as np

##2D Gan generator for alignments
class DC_Generator(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope):
        super(DC_Generator, self).__init__()
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, data_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(data_size * 8),
            nn.ReLU(True),
            # state size. (data_size*8) x 4 x 4
            nn.ConvTranspose2d(data_size * 8, data_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(data_size * 4),
            nn.ReLU(True),
            # state size. (data_size*4) x 8 x 8
            nn.ConvTranspose2d(data_size * 4, data_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(data_size * 2),
            nn.ReLU(True),
            # state size. (data_size*2) x 16 x 16
            nn.ConvTranspose2d(data_size * 2, data_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(data_size),
            nn.ReLU(True),
            # state size. (data_size) x 32 x 32
            nn.ConvTranspose2d(data_size, 1, 4, 2, 1, bias=False), 
            nn.Tanh()
            # state size. (nc) x 64 x 64 (nc = 1 here, defined after data size)
        )

    def forward(self, x):
        return self.layers(x)


# 2D GAN discriminator for alignments
class DC_Discriminator(nn.Module):
    def __init__(self, data_size, negative_slope):
        super(DC_Discriminator, self).__init__()
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.layers = nn.Sequential(
            nn.Conv2d(1, data_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (data_size) x 32 x 32
            nn.Conv2d(data_size, data_size * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (data_size*2) x 16 x 16
            nn.Conv2d(data_size * 2, data_size * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (data_size*4) x 8 x 8
            nn.Conv2d(data_size * 4, data_size * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1,data_size*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (data_size*8) x 4 x 4
            #nn.Conv2d(data_size * 8, 1, 4, 1, 0, bias=False), if using linear activation, remove final convolution (I think??)
            nn.Flatten(),
            nn.Linear(data_size * 8 * 4 * 4,1) #some examples (improved wgan-gp) have linear output but original paper does not, just ends on the final convolution
            #nn.Sigmoid() #nick had a sigmoid here and was using it for wgan, which is incorrect (actually idk)
            #original wgan
        )

    def forward(self, x):
        return self.layers(x)

# Vanilla Generator used in original paper
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


# Vanilla Discriminator used in original paper
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


# not currently used but just a class to hold both models in one
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        x = self.generator(x)
        return self.discriminator(x)


# ConditionalGAN Generator
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

    def forward(self, x, labels, use_cuda=False, device=None):
        labels = torch.tensor(np.eye(self.num_classes)[labels.cpu().numpy().reshape(-1)], dtype=torch.float)
        if use_cuda:
            labels = labels.to(device)
        input_ = torch.cat((x, labels), 1)
        return self.layers(input_)


# ConditionalGAN Discriminator
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
    def forward(self, x, labels, use_cuda=False, device=None):
        labels = torch.tensor(np.eye(self.num_classes)[labels.cpu().numpy().reshape(-1)], dtype=torch.float)
        if use_cuda:
            labels = labels.to(device)
        input_ = torch.cat((x, labels), 1)
        return self.layers(input_)




