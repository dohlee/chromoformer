# Prepare your own input data for Chromoformer

(WIP) This directory provides an automated pipeline to generate input data for Chromoformer training and evaluation.

## Step-by-step description of the pipeline

1. To run the pipeline, please prepare the following data:

    - Raw Histone ChIP-seq reads for each histone modification (in `fastq` format)
    - Reference genome sequences (in `fasta` format)
    - Gene annotation (in `gtf` format)
    - Gene expression profile (in `csv` format)
    - Promoter - pCRE mapping info (in `csv` format)

2. Configure `config.yaml` file to point the files prepared in step 1.

3. Run the pipeline using the following command at commandline:

```shell
snakemake -pr -j [NUM_CORES]
```