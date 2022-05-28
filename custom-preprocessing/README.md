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

## Helper scripts

**prepare_ensg2tss.py**

```
python scripts/prepare_ensg2tss.py \
--input result/04_metadata/gencode.vM1.annotation.gtf \
--output result/04_metadata/ensg2tss.pickle
```

**prepare_tss2fragment_hic.py**

```
python scripts/prepare_tss2fragment_hic.py \
--ensg2tss result/04_metadata/ensg2tss.pickle \
--fragment-size 40000 \
--output result/04_metadata/tss2fragment.pickle \
```

**prepare_frag2neighbors_and_pair2score.py**
```
python scripts/prepare_frag2neighbors_and_pair2score.py \
--freq-matrices result/05_interaction_freqs/nij.chr1 result/05_interaction_freqs/nij.chr2 result/05_interaction_freqs/nij.chr3 result/05_interaction_freqs/nij.chr4 \
result/05_interaction_freqs/nij.chr5 result/05_interaction_freqs/nij.chr6 result/05_interaction_freqs/nij.chr7 result/05_interaction_freqs/nij.chr8 \
result/05_interaction_freqs/nij.chr9 result/05_interaction_freqs/nij.chr10 result/05_interaction_freqs/nij.chr11 result/05_interaction_freqs/nij.chr12 \
result/05_interaction_freqs/nij.chr13 result/05_interaction_freqs/nij.chr14 result/05_interaction_freqs/nij.chr15 result/05_interaction_freqs/nij.chr16 \
result/05_interaction_freqs/nij.chr17 result/05_interaction_freqs/nij.chr18 result/05_interaction_freqs/nij.chr19 \
--chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 \
--tss2fragment result/04_metadata/tss2fragment.pickle \
--fragment-size 40000 \
--freq-threshold 10 \
--output-frag2neighbors result/04_metadata/frag2neighbors.pickle \
--output-pair2score result/04_metadata/pair2score.pickle
```

**prepare_npy_files.py**

```
python scripts/prepare_npy_files.py \
--ensg2tss result/04_metadata/ensg2tss.pickle \
--tss2fragment result/04_metadata/tss2fragment.pickle \
--frag2neighbors result/04_metadata/frag2neighbors.pickle \
--pair2score result/04_metadata/pair2score.pickle \
--npz-dir result/03_npz \
--out-dir result/05_npy \
--freq-threshold 10.0 \
-m H3K4me1 H3K4me3 H3K9me3 H3K27me3 H3K36me3 H3K27ac H3K9ac \
-c chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19
```