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
from gan import ConditionalGenerator, ConditionalDiscriminator
from torch.autograd import Variable
from data_processing import save_models, plot_losses, plot_pca, create_AGs
from data_loader import PopulationsDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_size", default="600", help="size of noise input")
    parser.add_argument("--negative_slope", default="0.01", help="alpha value for LeakyReLU")
    parser.add_argument("--gen_lr", default="1e-4", help="generator learning rate")
    parser.add_argument("--disc_lr", default="4e-4", help="discriminator learning rate")
    parser.add_argument("--epochs", default="10000")
    parser.add_argument("--ag_size", default="216", help="number of artificial genomes (haplotypes) to be created"
                                                         "if 0, then no genomes created")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_freq", default="0", help="save model every save_freq epochs") # zero means don't save
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--idir", default="populations")
    parser.add_argument("--odir", default="CGAN_output")
    parser.add_argument("--plot", action="store_true")
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


def main():

    args = parse_args()

    odir = args.odir
    latent_size = int(args.latent_size)
    negative_slope = float(args.negative_slope)
    g_learn = float(args.gen_lr)
    d_learn = float(args.disc_lr)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    ag_size = int(args.ag_size)
    use_cuda = args.use_cuda
    save_freq = int(args.save_freq)
    device = torch.device('cuda' if use_cuda else 'cpu')

    ifiles = [x for x in os.listdir(args.idir) if x.endswith(".csv")]
    n_classes = len(ifiles)

    df = pd.DataFrame()
    pop_dict = {}
    for i, file in enumerate(ifiles):
        pop_dict[file.split(".")[0].split("_")[0]] = i
        population = pd.read_csv(os.path.join(args.idir, file))
        population["label"] = file.split(".")[0].split("_")[0]
        df = df.append(population)

    data_size = df.shape[1] - 1
    mask = df.isin([2, 3])
    df = df[~mask]
    df = df.dropna()
    if len(df) > ag_size * 5:
        df = df.sample(n=ag_size * 5) # need to test what this multiple should be
        df.reset_index(inplace=True)
        del df["index"]
        labels = df["label"]
        del df["label"]
    data = torch.FloatTensor(df.values - np.random.uniform(0, 0.1, size=(df.shape[0], df.shape[1])))

    # Read input
    data_labels = np.array(list(map(lambda x: pop_dict[x], labels)))
    reversed_pop_dict = {value: key for (key, value) in pop_dict.items()}
    genomes_data = PopulationsDataset(data, data_labels)
    dataloader = DataLoader(dataset=genomes_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Make generator
    generator = ConditionalGenerator(data_size, latent_size, negative_slope, n_classes)

    # Make discriminator
    discriminator = ConditionalDiscriminator(data_size, negative_slope, n_classes)

    losses = []

    loss_fn = nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learn)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learn)

    # Training iteration
    for i in range(epochs):

        for j, (X_real, y_real) in enumerate(dataloader):

            real = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0))
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0))

            ### ----------------------------------------------------------------- ###
            #                           train generator                           #
            ### ----------------------------------------------------------------- ###
            gen_optimizer.zero_grad()
            z = torch.normal(0, 1, size=(batch_size, latent_size)).to(device)
            gen_labels = Variable(torch.LongTensor(np.random.randint(0, n_classes, batch_size)))

            # create batch of samples
            X_batch_fake = generator(z, gen_labels)

            # test samples
            y_pred = discriminator(X_batch_fake, gen_labels)
            gen_loss = loss_fn(y_pred, real)
            gen_loss.backward()
            gen_optimizer.step()

            ### ----------------------------------------------------------------- ###
            #                           train discriminator                       #
            ### ----------------------------------------------------------------- ###
            disc_optimizer.zero_grad()

            # train on real data
            real_preds = discriminator(X_real, y_real)
            disc_real_loss = loss_fn(real_preds, real - torch.FloatTensor(real.shape[0], real.shape[1]).uniform_(0, 0.1))

            # train on generated data
            fake_preds = discriminator(X_batch_fake.detach(), gen_labels)
            disc_fake_loss = loss_fn(fake_preds, fake)

            # total loss
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            disc_optimizer.step()

        losses.append((disc_loss.item(), gen_loss.item()))
        logging.info("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (i + 1, epochs, disc_loss.item(), gen_loss.item()))

        if save_freq != 0 and (i % save_freq == 1 or i == epochs):
            # Save models
            save_models(generator, discriminator, os.path.join(odir, "generator_model.pt"),
                        os.path.join(odir, "discriminator_model.pt"))

            if ag_size > 0:
                # Create AGs
                generated_genomes_df = create_AGs(generator, i, ag_size, latent_size, df, odir, reversed_pop_dict,
                                                  model_type="conditional")

                if args.plot:
                    plot_losses(odir, losses, i)

                    plot_pca(df, i, generated_genomes_df, odir, model_type="conditional", labels=labels)


if __name__ == "__main__":
    main()
