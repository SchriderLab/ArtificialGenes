# in the process of converting everything to pytorch

import sys
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
plt.switch_backend('agg')
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from gan import Generator, Discriminator, GAN

# in the process of converting everything to pytorch (way easier)

inpt = "../1000G_real_genomes/805_SNP_1000G_real.hapt" #hapt format input file
latent_size = 600 #size of noise input
negative_slope = 0.01 #alpha value for LeakyReLU
g_learn = 0.0001 #generator learning rate
d_learn = 0.0008 #discriminator learning rate
epochs = 10001
batch_size = 32
ag_size = 216 #number of artificial genomes (haplotypes) to be created
gpu_count = 0
lr = 1e-3
# gpu_count = 2 #number of GPUs
save_that = 1000 #epoch interval for saving outputs

#For saving models
def save_mod(gan, gen, disc, epo, save_gan_path, save_gen_path, save_disc_path):
    disc.eval() = False
    torch.save(gan.state_dict(), save_gan_path)
    disc.train()
    torch.save(gen.state_dict(), save_gen_path)
    torch.save(disc.state_dict(), save_disc_path)


#Read input
df = pd.read_csv(inpt, sep = ' ', header=None)
df = df.sample(frac=1).reset_index(drop=True)
df_noname = df.drop(df.columns[0:2], axis=1)
df_noname = df_noname.values
df_noname = df_noname - np.random.uniform(0,0.1, size=(df_noname.shape[0], df_noname.shape[1]))
df_noname = pd.DataFrame(df_noname)

#Make generator
generator = Generator(df_noname, latent_size, negative_slope)

#Make discriminator
discriminator = Discriminator(df_noname, negative_slope)

# if gpu_count > 1:
#     discriminator = multi_gpu_model(discriminator, gpus=gpu_count)

#Set discriminator to non-trainable
discriminator.eval()

#Make GAN
gan = GAN(generator, discriminator)

# if gpu_count > 1:
#     gan = multi_gpu_model(gan, gpus=gpu_count)

y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])
X_real = df_noname.values

losses = []
batches = len(X_real)//batch_size

loss_fn = nn.BCELoss()
gen_optimizer = torch.optimizers.Adam(generator.parameters(), lr=lr)
disc_optimizer = torch.optimizers.Adam(discriminator.parameters(), lr=lr)

#Training iteration
for i in range(epochs):
    for b in range(batches):
        X_batch_real = X_real[b*batch_size:(b+1)*batch_size] #get the batch from real data
        latent_samples = np.random.normal(loc=0, scale=1, size=(batch_size, latent_size)) #create noise to be fed to generator
        X_batch_fake = generator(latent_samples) #create batch from generator using noise as input

        #train discriminator, notice that noise is added to real labels
        discriminator.train()
        discriminator_real_preds = discriminator(X_batch_real)
        d_loss = loss_fn(discriminator_real_preds, y_real - np.random.uniform(0,0.1, size=(y_real.shape[0], y_real.shape[1])))
        discriminator_fake_preds = discriminator(X_batch_fake)
        d_loss += loss_fn(discriminator_fake_preds, y_fake)

        #make discriminator non-trainable and train gan
        discriminator.eval()

        ### haven't finished converting to pytorch yet
        g_loss = gan.train_on_batch(latent_samples, y_real)

    losses.append((d_loss, g_loss))
    print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f"%(i+1, epochs, d_loss, g_loss))
    if i % save_that == 0 or i == epochs:

        #Save models
        save_mod(gan, generator, discriminator, str(i))

        #Create AGs
        latent_samples = np.random.normal(loc=0, scale=1, size=(ag_size, latent_size))
        generator.eval()
        generated_genomes = generator(latent_samples)
        generated_genomes[generated_genomes < 0] = 0
        generated_genomes = np.rint(generated_genomes)
        generated_genomes_df = pd.DataFrame(generated_genomes)
        generated_genomes_df = generated_genomes_df.astype(int)
        gen_names = list()
        for j in range(0,len(generated_genomes_df)):
                gen_names.append('AG'+str(i))
        generated_genomes_df.insert(loc=0, column='Type', value="AG")
        generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
        generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
        df.columns = list(range(df.shape[1]))

        #Output AGs in hapt format
        generated_genomes_df.to_csv(str(i)+"_output.hapt", sep=" ", header=False, index=False)

        #Output lossess
        pd.DataFrame(losses).to_csv(str(i)+"_losses.txt", sep=" ", header=False, index=False)
        fig, ax = plt.subplots()
        plt.plot(np.array([losses]).T[0], label='Discriminator')
        plt.plot(np.array([losses]).T[1], label='Generator')
        plt.title("Training Losses")
        plt.legend()
        fig.savefig(str(i)+'_loss.pdf', format='pdf')

        #Make PCA
        df_pca = df.drop(df.columns[1], axis=1)
        df_pca.columns = list(range(df_pca.shape[1]))
        df_pca.iloc[:,0] = 'Real'
        generated_genomes_pca = generated_genomes_df.drop(generated_genomes_df.columns[1], axis=1)
        generated_genomes_pca.columns = list(range(df_pca.shape[1]))
        df_all_pca = pd.concat([df_pca, generated_genomes_pca])
        pca = PCA(n_components=2)
        PCs = pca.fit_transform(df_all_pca.drop(df_all_pca.columns[0], axis=1))
        PCs_df = pd.DataFrame(data = PCs, columns = ['PC1', 'PC2'])
        PCs_df['Pop'] = list(df_all_pca[0])
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        pops = ['Real', 'AG']
        colors = ['r', 'b']
        for pop, color in zip(pops,colors):
            ix = PCs_df['Pop'] == pop
            ax.scatter(PCs_df.loc[ix, 'PC1']
                       , PCs_df.loc[ix, 'PC2']
                       , c = color
                       , s = 50, alpha=0.2)
        ax.legend(pops)
        fig.savefig(str(i)+'_pca.pdf', format='pdf')

