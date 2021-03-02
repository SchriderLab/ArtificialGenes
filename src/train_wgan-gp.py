import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from collections import deque

plt.switch_backend('agg')

import torch
import torch.nn as nn
import argparse
from gan import Generator, Discriminator
from torch import autograd
from torch.autograd import Variable
from sklearn.decomposition import PCA


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
    parser.add_argument("--plot_pca", action="store_true")
    parser.add_argument("--plot_loss", action="store_true")
    parser.add_argument("--gpu_count", default="0")
    parser.add_argument("--critic_iter", default="5")
    parser.add_argument("--verbose", action="store_true")

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


def save_models(gen, disc, save_gen_path, save_disc_path):
    torch.save(gen.state_dict(), save_gen_path)
    torch.save(disc.state_dict(), save_disc_path)


def plot_losses(odir, losses):
    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator')
    plt.plot(np.array([losses]).T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    fig.savefig(os.path.join(odir, 'training_loss.pdf'), format='pdf')


# need to debug and test this gradient penalty once we get gradient clipping to work
def gradient_penalty(discriminator, real_batch, fake_batch, _lambda=10):
    t = torch.FloatTensor(real_batch.shape[0], 1).uniform_(0,1) # need to make sure shape is correct
    # t = t.expand(real_batch.shape[0], real_batch.shape[1])

    interpolated = t * real_batch + ((1-t) * fake_batch)
    # define as variable to calculate gradient
    interpolated = Variable(interpolated, requires_grad = True)

    # calculate probabilities of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * _lambda
    return grad_penalty


def plot_pca(df, generated_genomes_df, odir, i):
    df_pca = df.drop(df.columns[1], axis=1)
    df_pca.columns = list(range(df_pca.shape[1]))
    df_pca.iloc[:, 0] = 'Real'
    generated_genomes_pca = generated_genomes_df.drop(generated_genomes_df.columns[1], axis=1)
    generated_genomes_pca.columns = list(range(df_pca.shape[1]))
    df_all_pca = pd.concat([df_pca, generated_genomes_pca])
    pca = PCA(n_components=2)
    PCs = pca.fit_transform(df_all_pca.drop(df_all_pca.columns[0], axis=1))
    PCs_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
    PCs_df['Pop'] = list(df_all_pca[0])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    pops = ['Real', 'AG']
    colors = ['r', 'b']
    for pop, color in zip(pops, colors):
        ix = PCs_df['Pop'] == pop
        ax.scatter(PCs_df.loc[ix, 'PC1']
                   , PCs_df.loc[ix, 'PC2']
                   , c=color
                   , s=50, alpha=0.2)
    ax.legend(pops)
    fig.savefig(os.path.join(odir, str(i) + '_pca.pdf'), format='pdf')


def create_AGs(generator, i, ag_size, latent_size, df, odir):
    z = torch.normal(0, 1, size=(ag_size, latent_size))
    generator.eval()
    generated_genomes = generator(z).detach().numpy()
    generated_genomes[generated_genomes < 0] = 0
    generated_genomes = np.rint(generated_genomes)
    generated_genomes_df = pd.DataFrame(generated_genomes)
    generated_genomes_df = generated_genomes_df.astype(int)
    gen_names = list()
    for j in range(0, len(generated_genomes_df)):
        gen_names.append('AG' + str(i))
    generated_genomes_df.insert(loc=0, column='Type', value="AG")
    generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
    generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
    df.columns = list(range(df.shape[1]))

    # Output AGs in hapt format
    generated_genomes_df.to_csv(os.path.join(odir, str(i) + "_output.hapt"), sep=" ", header=False, index=False)

    # Output losses
    # pd.DataFrame(losses).to_csv(os.path.join(odir, str(i) + "_losses.txt"), sep=" ", header=False, index=False)
    return generated_genomes_df


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
    df = pd.read_csv(ifile, sep=' ', header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    df_noname = df.drop(df.columns[0:2], axis=1)
    df_noname = df_noname.values
    df_noname = df_noname - np.random.uniform(0, 0.1, size=(df_noname.shape[0], df_noname.shape[1]))
    df_noname = pd.DataFrame(df_noname)

    # Make generator
    generator = Generator(df_noname, latent_size, negative_slope).to(device) #

    # Make discriminator
    discriminator = Discriminator(df_noname, negative_slope).to(device) #

    # if gpu_count > 1:
    #     discriminator = multi_gpu_model(discriminator, gpus=gpu_count)

    X_real = df_noname.values

    # losses = deque(maxlen=1000)
    losses = []
    batches = len(X_real) // batch_size

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learn, betas=(beta1, beta2))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learn, betas=(beta1, beta2))

    one = torch.tensor(1, dtype=torch.float).to(device) #
    neg_one = one * -1
    neg_one = neg_one.to(device)

    # Training iteration
    for i in range(epochs):

        for p in discriminator.parameters():
            p.requires_grad = True

        d_loss_real = 0
        d_loss_fake = 0
        wasserstein_d = 0

        for j, _ in enumerate(range(critic_iter)):


            ### ----------------------------------------------------------------- ###
            #                           train critic                                #
            ### ----------------------------------------------------------------- ###
            discriminator.train()
            disc_optimizer.zero_grad()
            generator.eval()

            # train with real images
            X_batch_real = torch.FloatTensor(X_real[j * batch_size:(j + 1) * batch_size]).to(device)
            d_loss_real = discriminator(X_batch_real)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(neg_one)

            # train with fake images
            z = Variable(torch.normal(0, 1, size=(batch_size, latent_size))).to(device)
            X_batch_fake = generator(z).detach().to(device)
            d_loss_fake = discriminator(X_batch_fake)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)

            # for gradient penalty
            g_penalty = gradient_penalty(discriminator, X_batch_real, X_batch_fake)
            g_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + g_penalty

            wasserstein_d += d_loss_real - d_loss_fake
            disc_optimizer.step()


        ### ----------------------------------------------------------------- ###
        #                           train generator                           #
        ### ----------------------------------------------------------------- ###
        for p in discriminator.parameters():
            p.requires_grad = False
        generator.train()
        gen_optimizer.zero_grad()
        discriminator.eval()

        z = Variable(torch.normal(0, 1, size=(batch_size, latent_size))).to(device)
        X_batch_fake = generator(z)
        gen_loss = discriminator(X_batch_fake)
        # print(gen_loss)
        # print(gen_loss.mean())
        gen_loss = gen_loss.mean() # see if you need .mean(0) or something
        gen_loss.backward(neg_one)
        gen_optimizer.step()

        losses.append((wasserstein_d.mean().item(), gen_loss.item()))
        logging.info("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (i + 1, epochs, wasserstein_d.mean().item(), gen_loss.item()))

        if save_freq != 0 and (i % save_freq == 1 or i == epochs):
            # Save models
            save_models(generator, discriminator, os.path.join(odir, "generator_model.pt"),
                        os.path.join(odir, "discriminator_model.pt"))

            if ag_size > 0:
                # Create AGs
                generated_genomes_df = create_AGs(generator, i, ag_size, latent_size, df, odir)

            if args.plot_pca:
                plot_pca(df, generated_genomes_df, odir, i)

    if args.plot_loss:
        plot_losses(odir, losses)


if __name__ == "__main__":
    main()