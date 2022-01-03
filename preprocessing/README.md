# Data preprocessing pipeline

We provide the whole data preprocessing pipeline used in this study as a single snakemake pipeline.

Running this pipeline will automatically 1) download all the relevant tagAlign files from Roadmap Epigenomics (for the 11 cell types analyzed in the study), 2) convert the tagAlign files into BAM files using `bedtools bedtobam`, 3) save the read depths throughout the hg19 reference genome as bedGraph files using `bedtools genomecov` and 4) convert the read depth signal to npz files. Using npz file makes it easy to extract histone read depth signals in the form of numpy array.

### Quickstart

Install [conda](https://docs.conda.io/en/latest/) package manager and [snakemake](https://snakemake.readthedocs.io/en/stable/) workflow managing system, and run the pipeline using the command below:

```
snakemake --resources network=1 -j32 --use-conda --conda-frontend mamba -pr
```

Adjust the number of `network` resource according to your network bandwidth and I/O capacity (e.g., increasing it to 2 will allow at most two concurrent downloads). You can also adjust the number of processors to use with the parameter `-j`. Note that `--use-conda` option will automatically install the required tools and python packages for you. We highly recommend to install `mamba`, which remarkably reduces the time to create environment compared to `conda`. If you do not have `mamba` installed in your system, install it in the base environment with:

```
conda install mamba -n base -c conda-forge
```

### Description

This pipeline downloads ChIP-seq data (in tagAlign files) from [Roadmap Epigenomics Web Portal](https://egg2.wustl.edu/roadmap/web_portal/) and process them into a bunch of numpy matrix files (in .npy) so that it can be readily used for the training procedure.

Only 11 out of 127 reference epigenomcs provided in the portal were selected for this study since they were sequenced for seven major histone marks (H3K4me1, H3K4me3, H3K9me3, H3K27me3, H3K36me3, H3K27ac, H3K9ac) of our interest. Indeed, it will be interesting to train Chromoformer with the whole set of histone marks including those seven and see if there is any performance boost, but we leave it for further study.

The whole procedure for preprocessing is as below:

1. Download histone ChIP-seq tagAlign files from Roadmap (`wget`)
2. Convert tagAlign files into BAM files, and sort/index them (`bedtools bedtobam`, `sambamba sort/index`)
3. Compute genomewide read depth using BAM files and save it as bedGraph files (`bedtools genomecov`)
4. Convert bedGraph in to npz files for easy file handling (`bdg2npz.py`)

### Note
You may need about ~400G of disk space to save all the raw histone signal data (tagAlign, bedGraph, and npz files).

