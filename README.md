# ArtificialGenes
Using generative networks to create sequences of artificial DNA

Steps: 
   1) Duplicate results of initial paper (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009303)
   2) Try to generate better samples using Wasserstein GANs
   3) Use controllable and conditional GANs to generate specific types of sequences
     (ex. generate sequences from a specific population or with a certain haplotype)
   4) Test Sparse Transformers for long sequences of DNA
     (https://arxiv.org/abs/1904.10509) 
    
  __________________________________________________________________________________________________________________________________________________________________
  
  Resources: 
  
  https://github.com/openai/blocksparse:
    
    pip install blocksparse
    
  https://github.com/openai/sparse_attention/blob/master/attention.py

  https://github.com/openai/distribution_augmentation
  
  Disclaimer: 
  Some files in the "1000G_real_genomes" directory were taken directly from the code corresponding to this paper:
  https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009303

