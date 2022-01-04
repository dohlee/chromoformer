# Chromoformer

This repository provides the official code implementations for Chromoformer.

We also provide our pipelines for preprocessing input data and training Chromoformer model to help researchers reproduce the results and extend the study with their own data.
The repository includes two directories: `preprocessing`, `chromoformer`.

Refer to the directory named [`preprocessing`](preprocessing) to explore how we preprocessed the ChIP-seq signals and gene expression data for 11 cell lines from Roadmap Epigenomics. We provide the whole preprocessing pipeline as a one-shot `snakemake` pipeline. One can easily extend the pipeline for other cell types or other histone marks by slightly tweaking the parameters.

[`chromoformer`](chromoformer) directory provides the PyTorch implementation of the Chromoformer model.

## Model description

## Citation

Lee, D., Yang, J., & Kim, S. (2021). Learning the histone codes of gene regulation with large genomic windows and three-dimensional chromatin interactions using transformer. bioRxiv.

```
@article{lee2021learning,
  title={Learning the histone codes of gene regulation with large genomic windows and three-dimensional chromatin interactions using transformer},
  author={Lee, Dohoon and Yang, Jeewon and Kim, Sun},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
