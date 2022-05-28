import pandas as pd
from pathlib import Path

wildcard_constraints:
    run = 'ENCFF[0-9]+[A-Z]+',

configfile: 'config.yaml'
manifest = pd.read_csv(config['manifest'])
data_dir = Path(config['data_dir'])

bam_runs = manifest[manifest.file_type == 'bam'].accession.values
fastq_single_runs = manifest[manifest.file_type == 'fastq_single'].accession.values

ALL = []
ALL.append(expand(str(data_dir / '{run}.bam'), run=bam_runs))
ALL.append(expand(str(data_dir / '{run}.fastq.gz'), run=fastq_single_runs))

rule all:
    input: ALL

rule download_bam_from_ENCODE:
    output:
        data_dir / '{run}.bam'
    resources:
        network = 1
    shell:
        'wget https://www.encodeproject.org/files/{wildcards.run}/@@download/{wildcards.run}.bam -O {output}'

rule download_fastq_single_from_ENCODE:
    output:
        data_dir / '{run}.fastq.gz'
    resources:
        network = 1
    shell:
        'wget https://www.encodeproject.org/files/{wildcards.run}/@@download/{wildcards.run}.fastq.gz -O {output}'
