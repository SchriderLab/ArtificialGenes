# ArtificialGenes
* This repository contains code for using generative deep learning to create sequences of artificial DNA

* The work in this repository is expanding on the work in the paper below and reuses some code:
   https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009303

To run the original vanilla GAN from that paper:

``` slurm/tf_gan.sh ```

To run a pytorch implementation of the vanilla GAN:

``` slurm/run_gan.sh input_file output_dir 200 ```

To run a Wasserstein GAN with Gradient Penalty:

``` slurm/run_wgan.sh input_file output_dir 200 ```

To run a Conditional GAN:

``` slurm/run_cgan.sh populations populations_output 200 ```

To parse data from the 1000 genomes dataset (consult script for all parsing options):

``` python3 src/vcf_data_extractor.py --chrom_list 14,15,16 --odir multi_chrom_output --population YRI ```

To convert simulated data to .csv data:

``` python3 src/simulated_data_extractor.py --idir simulated_data --odir unzipped_data ```

To format simulated .csv data to your preferred specifications:

``` python3 src/simulated_data_processing.py --idir unzipped_data --length 1000 --ofile full_simulated_data.csv ```

Ideas for next steps:
1) Find highly divergent data and train a CGAN to successful generate samples of each class
2) Train Controllable GAN to convert input sequence dataset to dataset with selective sweep
3) Use Transformer model to convert input sequence data to data with selective sweep
4) Explore Progressive Growing as a strategy to stably generate long, high-quality sequence data (this might not work)
5) Use Sparse Transformers to generate extremely long sequence data 
    
  __________________________________________________________________________________________________________________________________________________________________

 
  Disclaimer: 
  Some files in the "1000G_real_genomes" directory were taken directly from the code corresponding to this paper:
  https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009303

