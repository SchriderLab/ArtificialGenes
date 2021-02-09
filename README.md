# ArtificialGenes
Using sparse transformers and other generative models to generate long DNA sequences

Initial Idea:

  Building on this paper: Creating Artificial Human Genomes Using Generative Models (https://www.biorxiv.org/content/10.1101/769091v2.full.pdf), we're 
  trying to use a number of different generative models to generative realistic long DNA sequences. 
  
  We're especially interested in the potential
  of Sparse Transformers, introduced here: https://arxiv.org/abs/1904.10509 which are generative transformer models especially good at long-range 
  sequence generation. 
  
  We'll start with a single population from the 1000 Genomes Project.
  
  __________________________________________________________________________________________________________________________________________________________________
  
  Resources: 
  
  https://github.com/openai/blocksparse:
    
    pip install blocksparse
    
  https://github.com/openai/sparse_attention/blob/master/attention.py

  https://github.com/openai/distribution_augmentation
  
  Disclaimer: 
  All files in the "1000G_real_genomes" directory were taken directly from the code corresponding to this paper:
  https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009303

