## Data preprocessing pipeline

We provide the whole data preprocessing pipeline used in this study as a single snakemake pipeline.

Running this pipeline will automatically 1) download all the relevant tagAlign files from Roadmap Epigenomics (for the 11 cell types analyzed in the study), 2) convert the tagAlign files into BAM files using `bedtools bedtobam`, 3) save the read depths throughout the hg19 reference genome as bedGraph files using `bedtools genomecov` and 4) convert the read depth signal to npz files. Using npz file makes it easy to extract histone read depth signals in the form of numpy array.

### Run the pipeline

Adjust the number of `network` resource according to your network bandwidth and I/O capacity.

Also you can adjust the number of processors to use with the parameter `-j`.

```
snakemake --resources network=1 -j32 --use-conda -pr
```

### Note
You may need about ~400G of disk space to save all the raw histone signal data (tagAlign, bedGraph, and npz files).

