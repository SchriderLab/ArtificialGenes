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
from torch.autograd import Variable
from sklearn.decomposition import PCA
from data_loader import GenomesDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_size", default="600", help="size of noise input")
    parser.add_argument("--negative_slope", default="0.01", help="alpha value for LeakyReLU")
    parser.add_argument("--gen_lr", default="1e-4", help="generator learning rate")
    parser.add_argument("--disc_lr", default="8e-4", help="discriminator learning rate")
    parser.add_argument("--epochs", default="10000")
    parser.add_argument("--ag_size", default="216", help="number of artificial genomes (haplotypes) to be created"
                                                         "if 0, then no genomes created")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_freq", default="0", help="save model every save_freq epochs") # zero means don't save
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--ifile", default="../1000G_real_genomes/805_SNP_1000G_real.hapt")
    parser.add_argument("--odir", default="../output")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu_count", default="0")
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


def plot_losses(odir, losses, i):
    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator')
    plt.plot(np.array([losses]).T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    fig.savefig(os.path.join(odir, str(i) + '_loss.pdf'), format='pdf')


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
    # generated_genomes_df.insert(loc=0, column='Type', value="AG")
    # generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
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
    gpu_count = int(args.gpu_count)
    save_freq = int(args.save_freq)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Read input
    genomes_data = GenomesDataset(ifile)
    dataloader = DataLoader(dataset=genomes_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # dataiter = iter(dataloader)
    if ".hapt" in ifile:
        data = pd.read_csv(ifile, sep=' ', header=None)
        data = data.reset_index(drop=True)
        data = data.drop(data.columns[0:2], axis=1).values
        data_size = 805
    else:
        data = pd.read_csv(ifile)
        # test just sequence from rows 7k-8k
        # data = data.iloc[7000:7805, :].T
        df = data.reset_index(drop=True)
        data = df.values
        data_size = 1000 # temp changed from 1000
    data = torch.FloatTensor(data - np.random.uniform(0, 0.1, size=(data.shape[0], data.shape[1])))

    # Make generator
    generator = Generator(data_size, latent_size, negative_slope)

    # Make discriminator
    discriminator = Discriminator(data_size, negative_slope)

    # if gpu_count > 1:
    #     discriminator = multi_gpu_model(discriminator, gpus=gpu_count)

    # losses = deque(maxlen=1000)
    losses = []
    # batches = len(dataloader)

    loss_fn = nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learn)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learn)

    # Training iteration
    for i in range(epochs):

        disc_losses = []
        gen_losses = []

        for j, X_real in enumerate(dataloader):

            ones = Variable(torch.Tensor(batch_size, 1).fill_(1).type(torch.FloatTensor))
            zeros = Variable(torch.Tensor(batch_size, 1).fill_(0).type(torch.FloatTensor))

            # get the batch from real data
            z = torch.normal(0, 1, size=(batch_size, latent_size)).to(device)
            X_fake = generator(z).detach().to(device)  # create batch from generator using noise as input

            ### ----------------------------------------------------------------- ###
            #                           train discriminator                       #
            ### ----------------------------------------------------------------- ###
            discriminator.train()
            disc_optimizer.zero_grad()
            generator.eval()
            real_preds = discriminator(X_real)
            disc_loss = loss_fn(real_preds, ones - torch.FloatTensor(ones.shape[0], ones.shape[1]).uniform_(0, 0.1))
            fake_preds = discriminator(X_fake)
            disc_loss += loss_fn(fake_preds, zeros)
            disc_loss.backward()
            disc_optimizer.step()

            ### ----------------------------------------------------------------- ###
            #                           train generator                           #
            ### ----------------------------------------------------------------- ###
            generator.train()
            gen_optimizer.zero_grad()
            discriminator.eval()
            z = torch.normal(0, 1, size=(batch_size, latent_size)).to(device)
            X_batch_fake = generator(z)
            y_pred = discriminator(X_batch_fake)
            gen_loss = loss_fn(y_pred, ones)
            gen_loss.backward()
            gen_optimizer.step()

            losses.append((disc_loss.item(), gen_loss.item()))
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())
        logging.info("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (i + 1, epochs, np.mean(disc_losses), np.mean(gen_losses)))

        if save_freq != 0 and (i % save_freq == 1 or i == epochs):
            # Save models
            save_models(generator, discriminator, os.path.join(odir, "generator_model.pt"),
                        os.path.join(odir, "discriminator_model.pt"))

            if ag_size > 0:
                # Create AGs
                generated_genomes_df = create_AGs(generator, i, ag_size, latent_size, df, odir)

                if args.plot:
                    plot_losses(odir, losses, i)

                    plot_pca(df, generated_genomes_df, odir, i)


if __name__ == "__main__":
    main()
