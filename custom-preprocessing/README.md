# Prepare your own input data for Chromoformer

(WIP) This directory provides an automated pipeline to generate the whole input data for Chromoformer training and evaluation.
Currently there are two ways to generate input data.

1. Starting from raw histone ChIP-seq reads. Reads will be aligned to reference genome, and read depth normalization and calculation will be done. **(See 1. FASTQ pipeline)**
2. Starting from pre-aligned reads (from pre-aligned BAM files). **(See 2. BAM pipeline)**

## 1. Step-by-step description of the FASTQ pipeline

1. To run the pipeline, please prepare the following data:

    - Raw Histone ChIP-seq reads for each histone modification (in `fastq` format)
    - Reference genome sequences (in `fasta` format)
    - Gene annotation (in `gtf` format)
    - Gene expression profile (in `csv` format)
    - Promoter - pCRE mapping info (in `csv` format)

2. Configure `config.yaml` file to point the files prepared in step 1.

3. Run the pipeline using the following command at commandline:

```shell
snakemake -s fastq_pipeline.smk -pr -j [NUM_CORES]
```

## 2. Step-by-step description of the BAM pipeline

1. To run the pipeline, please prepare the following data:

    - Aligned reads (in `bam` format)
    - Reference genome sequences (in `fasta` format)
    - Gene annotation (in `gtf` format)
    - Gene expression profile (in `csv` format)
    - Promoter - pCRE mapping info (in `csv` format)

2. Configure `config.yaml` file to point the files prepared in step 1.

3. Run the pipeline using the following command at commandline:

```shell
snakemake -s bam_pipeline.smk -pr -j [NUM_CORES]
```

**Using helper pipeline for downloading files from ENCODE**

We provide a convenient pipeline named `ENCODE_download_helper.smk` to help users easily fetch aligned histone ChIP-seq data from ENCODE.
Before running the pipeline, you should prepare a `manifest.csv` table that specifies the files deposited in ENCODE database that you are going to use.
The required columns for `manifest.csv` file are as below:

- accession : ENCODE file accession (ENCFF481UTW)
- mark : Histone modification targeted by ChIP-seq (H3K4me1 / H3K4me3 / ...)
- file_type : Corresponding file type (bam, fastq_single, fastq_paired)

After preparing `manifest.csv` file, just run the pipeline to fetch all the files needed.
Adjust `--resources network=N` parameter depending on your network status.
It controls the maximum number of concurrent downloads.

```shell
snakemake -s ENCODE_download_helper.smk -pr --resources network=1 -j [NUM_CORES]
```
