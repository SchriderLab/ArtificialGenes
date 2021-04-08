import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from data_loader import GenomesDataset
from torch.utils.data import DataLoader
from data_processing import save_models, plot_losses, plot_pca, create_AGs

plt.switch_backend('agg')

import torch
import argparse
from gan import Generator, Discriminator
from torch import autograd
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_size", default="600", help="size of noise input")
    parser.add_argument("--negative_slope", default="0.01", help="alpha value for LeakyReLU")
    parser.add_argument("--gen_lr", default="1e-4", help="generator learning rate")
    parser.add_argument("--disc_lr", default="1e-4", help="discriminator learning rate")
    parser.add_argument("--epochs", default="10000")
    parser.add_argument("--ag_size", default="216", help="number of artificial genomes (haplotypes) to be created"
                                                         "if 0, then no genomes created")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_freq", default="0", help="save model every save_freq epochs") # zero means don't save
    parser.add_argument("--batch_size", default="64")
    parser.add_argument("--ifile", default="../1000G_real_genomes/805_SNP_1000G_real.hapt")
    parser.add_argument("--odir", default="../output")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu_count", default="0")
    parser.add_argument("--critic_iter", default="5")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--data_size", default="1000")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args


def gradient_penalty(discriminator, real_batch, fake_batch, _lambda=10):
    t = torch.FloatTensor(real_batch.shape[0], 1).uniform_(0,1)

    interpolated = t * real_batch + ((1-t) * fake_batch)
    # define as variable to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probabilities of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * _lambda
    return grad_penalty


def main():

    args = parse_args()

    ifile = args.ifile
    odir = args.odir
    latent_size = int(args.latent_size)
    negative_slope = float(args.negative_slope)
    g_learn = float(args.gen_lr)
    d_learn = float(args.disc_lr)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    ag_size = int(args.ag_size)
    use_cuda = args.use_cuda
    gpu_count = int(args.gpu_count) # not used atm
    save_freq = int(args.save_freq)
    critic_iter = int(args.critic_iter)
    beta1 = 0.5
    beta2 = 0.999
    lambda_term = 10

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Read input
    genomes_data = GenomesDataset(ifile)
    dataloader = DataLoader(dataset=genomes_data, batch_size=batch_size, shuffle=True, drop_last=True)

    if ".hapt" in ifile:
        df = pd.read_csv(ifile, sep=' ', header=None)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.drop(df.columns[0:2], axis=1)
        df = df.values
        data_size = 805
    else:
        data = pd.read_csv(ifile)
        data = data.values
        data_size = int(args.data_size)
        df = pd.DataFrame(data)
    data = torch.FloatTensor(data - np.random.uniform(0, 0.1, size=(data.shape[0], data.shape[1])))

    # Make generator
    generator = Generator(data_size, latent_size, negative_slope).to(device) #

    # Make discriminator
    discriminator = Discriminator(data_size, negative_slope).to(device) #

    # if gpu_count > 1:
    #     discriminator = multi_gpu_model(discriminator, gpus=gpu_count)

    losses = []

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learn, betas=(beta1, beta2))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learn, betas=(beta1, beta2))

    one = torch.tensor(1, dtype=torch.float).to(device)
    neg_one = one * -1
    neg_one = neg_one.to(device)

    epoch_length = len(iter(dataloader))

    # Training iteration
    for i in range(epochs):

        for j, X_real in enumerate(dataloader):

            for p in discriminator.parameters():
                p.requires_grad = True

            wasserstein_d = 0

            for h in range(critic_iter):


                ### ----------------------------------------------------------------- ###
                #                           train critic                                #
                ### ----------------------------------------------------------------- ###
                disc_optimizer.zero_grad()
                # train with real images
                d_loss_real = discriminator(X_real)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(neg_one)

                # train with fake images
                z = Variable(torch.normal(0, 1, size=(batch_size, latent_size))).to(device)
                X_batch_fake = generator(z).detach().to(device)
                d_loss_fake = discriminator(X_batch_fake)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # for gradient penalty
                g_penalty = gradient_penalty(discriminator, X_real, X_batch_fake)
                g_penalty.backward()

                # d_loss = d_loss_fake - d_loss_real + g_penalty

                wasserstein_d += d_loss_real - d_loss_fake
                disc_optimizer.step()


            ### ----------------------------------------------------------------- ###
            #                           train generator                           #
            ### ----------------------------------------------------------------- ###
            for p in discriminator.parameters():
                p.requires_grad = False
            gen_optimizer.zero_grad()

            z = Variable(torch.normal(0, 1, size=(batch_size, latent_size))).to(device)
            X_batch_fake = generator(z)
            gen_loss = discriminator(X_batch_fake)
            gen_loss = gen_loss.mean()
            gen_loss.backward(neg_one)
            gen_optimizer.step()

            if j % epoch_length - 1 == 0 and j != 0:
                losses.append((wasserstein_d.mean().item(), gen_loss.item()))
                logging.info("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (i + 1, epochs, wasserstein_d.mean().item(), gen_loss.item()))

        if save_freq != 0 and (i % save_freq == 1 or i == epochs):
            # Save models
            save_models(generator, discriminator, os.path.join(odir, "generator_model.pt"),
                        os.path.join(odir, "discriminator_model.pt"))

            if ag_size > 0:
                # Create AGs
                generated_genomes_df = create_AGs(generator, ifile, i, ag_size, latent_size, df, odir)

                if args.plot:
                    plot_losses(odir, losses, i)

                    plot_pca(df, ifile, generated_genomes_df, odir, i)


if __name__ == "__main__":
    main()
