import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd


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
    plt.close(fig)


def plot_pca(df, generated_genomes_df, odir, i, pop_dict=None, model_type="normal"):


    df_temp = df.copy()
    # df_temp["label"] = df_temp["label"].map(lambda x: "real_"+x)
    # need to change this to separate the different populations in a conditional GAN
    df_temp["label"] = "Real"
    generated_genomes_df['label'] = "AG"
    df_all_pca = pd.concat([df_temp, generated_genomes_df])
    pca = PCA(n_components=2)
    labels = df_all_pca.pop("label").to_list()
    PCs = pca.fit_transform(df_all_pca)
    PCs_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
    PCs_df['Pop'] = labels

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
    plt.cla()
    plt.close(fig)


def create_AGs(generator, i, ag_size, latent_size, df, odir, pop_dict=None, model_type="normal"):

    def pop_map(value, pop_dict=pop_dict):
        return pop_dict[value]

    z = torch.normal(0, 1, size=(ag_size, latent_size))
    generator.eval()
    if model_type == "conditional":
        classes = torch.randint(0, 2, (ag_size,))
        generated_genomes = generator(z, classes).detach().numpy()
    else:
        generated_genomes = generator(z).detach().numpy()
    generated_genomes[generated_genomes < 0] = 0
    generated_genomes = np.rint(generated_genomes)
    generated_genomes_df = pd.DataFrame(generated_genomes)
    generated_genomes_df = generated_genomes_df.astype(int)
    generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))

    if model_type == "conditional":
        generated_genomes_df["label"] = classes.numpy()
        generated_genomes_df["label"] = generated_genomes_df["label"].map(lambda x: pop_dict[x])
    df.columns = list(range(df.shape[1]))

    generated_genomes_df.to_csv(os.path.join(odir, str(i) + "_output.hapt"), sep=" ", header=False, index=False)

    return generated_genomes_df