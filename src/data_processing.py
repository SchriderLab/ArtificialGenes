import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
from torchvision.utils import make_grid, save_image


# saves models
def save_models(gen, disc, save_gen_path, save_disc_path):
    torch.save(gen.state_dict(), save_gen_path)
    torch.save(disc.state_dict(), save_disc_path)


# plots and records losses for entire training time
def plot_losses(odir, losses, i):
    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator')
    plt.plot(np.array([losses]).T[1], label='Generator')
    plt.plot(np.array([losses]).T[2], label='D_loss_real')
    plt.plot(np.array([losses]).T[3], label='D_loss_fake')
    plt.plot(np.array([losses]).T[4], label='Gradient_Penalty')
    
    plt.title("Training Losses")
    plt.legend()
    fig.savefig(os.path.join(odir, str(i) + '_loss.png'), format='png')
    plt.close(fig)


# plots and records pca comparing real and generated sequences
def plot_pca(df, i, generated_genomes_df, odir, model_type="normal", labels=None, pop_dict=None, projection="2d"):

    # copy original data
    df_temp = df.copy()

    # set up data labels for plotting
    if model_type == "conditional":
        pop_labels = list(pop_dict.keys())
        pops = ["Real_" + x for x in pop_labels]
        pops.extend(["AG_" + x for x in pop_labels])
        df_temp["label"] = labels.map(lambda x: "Real_" + x)
        generated_genomes_df['label'] = generated_genomes_df.loc[:, "label"].map(lambda x: "AG_" + x)
    else:
        df_temp["label"] = "Real"
        pops = ['Real', 'AG']

    # append Real and AG data
    df_all_pca = pd.concat([df_temp, generated_genomes_df])

    # calculate principal components and collect labels
    n_components = int(projection[0])
    pca = PCA(n_components=n_components)
    labels = df_all_pca.pop("label").to_list()
    PCs = pca.fit_transform(df_all_pca)

    fig = plt.figure(figsize=(10, 10))

    if projection == "2d":
        PCs_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    else:
        PCs_df = pd.DataFrame(data=PCs, columns=["PC1", "PC2", "PC3"])
        ax = plt.axes(projection="3d")

    PCs_df['Pop'] = labels

    # plot data
    for pop in pops:
        ix = PCs_df['Pop'] == pop
        if projection == "2d":
            ax.scatter(PCs_df.loc[ix, 'PC1']
                       , PCs_df.loc[ix, 'PC2']
                       , s=50, alpha=0.2)
        else:
            ax.scatter3D(PCs_df.loc[ix, 'PC1'], PCs_df.loc[ix, 'PC2'], PCs_df.loc[ix, 'PC3'], s=50, alpha=0.2)

    ax.legend(pops)
    fig.savefig(os.path.join(odir, str(i) + '_pca.pdf'), format='pdf')
    plt.cla()
    plt.close(fig)


def create_AGs(generator, i, ag_size, latent_size, data_size, odir, device):

    z = torch.randn(ag_size, latent_size, 1, 1, device=device)
    generator.eval()

    generated_genomes = generator(z).detach().cpu().numpy()
    generated_genomes[generated_genomes < 0] = 0
    generated_genomes = np.rint(generated_genomes)
    #generated_genomes_df = pd.DataFrame(generated_genomes)
    #generated_genomes = generated_genomes.astype(int)
    generated_genomes_3d = np.reshape(generated_genomes,(ag_size,data_size,data_size))
    #generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
    generated_genomes_imgs = save_image(torch.tensor(generated_genomes), os.path.join(odir, str(i) + "_output.png"))
#    if model_type == "conditional":
#        generated_genomes_df["label"] = classes.detach().cpu().numpy()
#        generated_genomes_df["label"] = generated_genomes_df["label"].map(lambda x: reversed_pop_dict[x])
#    else:
#        generated_genomes_df["label"] = "AG"
#    df.columns = list(range(df.shape[1]))
#
    #generated_genomes_df.to_csv(os.path.join(odir, str(i) + "_output.hapt"), sep=" ", header=False, index=False)
#
    return generated_genomes_imgs

# def create_AGs(generator, i, ag_size, latent_size, df, odir, reversed_pop_dict=None, model_type="normal",
#               n_classes=None, use_cuda=False, device=None):
#
#    z = torch.normal(0, 1, size=(ag_size, latent_size)).to(device)
#    generator.eval()
#    if model_type == "conditional":
#        classes = torch.randint(0, n_classes, (ag_size,)).to(device)
#        if use_cuda:
#           generated_genomes = generator(z, classes, use_cuda, device)
#        else:
#            generated_genomes = generator(z, classes)
#        generated_genomes = generated_genomes.detach().cpu().numpy()
#    else:
#        generated_genomes = generator(z).detach().cpu().numpy()
#    generated_genomes[generated_genomes < 0] = 0
#    generated_genomes = np.rint(generated_genomes)
#    generated_genomes_df = pd.DataFrame(generated_genomes)
#    generated_genomes_df = generated_genomes_df.astype(int)
#    generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
#
#    if model_type == "conditional":
#        generated_genomes_df["label"] = classes.detach().cpu().numpy()
#        generated_genomes_df["label"] = generated_genomes_df["label"].map(lambda x: reversed_pop_dict[x])
#    else:
#        generated_genomes_df["label"] = "AG"
#    df.columns = list(range(df.shape[1]))
#
#    generated_genomes_df.to_csv(os.path.join(odir, str(i) + "_output.hapt"), sep=" ", header=False, index=False)
#
#    return generated_genomes_df