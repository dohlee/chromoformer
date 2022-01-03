import pandas as pd

wildcard_constraints:
    eid = '[^.]+',
    mark = '[^.]+',

hg19_sizes = 'hg19.fa.sizes'

eids = [
    'E003', 'E004', 'E005', 'E006', 'E007',
    'E016', 'E066', 'E087', 'E114', 'E116',
    'E118',
]

marks = [
    'H3K4me1',
    'H3K4me3',
    'H3K9me3',
    'H3K27me3',
    'H3K36me3',
    'H3K27ac',
    'H3K9ac',
]

eids = ['E003']
marks = ['H3K4me1']

# Register target files. 
tagalign = expand('result/{eid}-{mark}.sorted.bam.bai', eid=eids, mark=marks)
coverage = expand('result/{eid}-{mark}.sorted.bedGraph', eid=eids, mark=marks)
npz = expand('result/{eid}-{mark}.npz', eid=eids, mark=marks)

ALL = []
ALL.append(tagalign)
ALL.append(coverage)
ALL.append(npz)

rule all:
    input: ALL

rule download:
    output: temp('result/{eid}-{mark}.tagAlign')
    resources: network = 1
    conda:
        'environment.yaml'
    shell:
        'wget "https://egg2.wustl.edu/roadmap/data/byFileType/alignments/consolidated/{wildcards.eid}-{wildcards.mark}.tagAlign.gz" -O- | gunzip -c > {output}'

rule bedtobam:
    input: 'result/{eid}-{mark}.tagAlign'
    output: temp('result/{eid}-{mark}.bam')
    conda:
        'environment.yaml'
    shell:
        'bedtools bedtobam -i {input} -g {hg19_sizes} > {output}'

rule sambamba_sort:
    input: 'result/{eid}-{mark}.bam'
    output: 'result/{eid}-{mark}.sorted.bam'
    threads: 4
    conda:
        'environment.yaml'
    shell:
        'sambamba sort -o {output} -t {threads} --tmpdir . {input}'

rule sambamba_index:
    input: 'result/{eid}-{mark}.sorted.bam'
    output: 'result/{eid}-{mark}.sorted.bam.bai'
    threads: 4
    conda:
        'environment.yaml'
    shell:
        'sambamba index -t {threads} {input}'

rule bedtools_genomecov:
    input:
        bam = 'result/{eid}-{mark}.sorted.bam',
        bai = 'result/{eid}-{mark}.sorted.bam.bai',
    output: 'result/{eid}-{mark}.sorted.bedGraph'
    conda:
        'environment.yaml'
    shell:
        'bedtools genomecov -ibam '
        '{input.bam} -bga | '
        'bedtools sort -i stdin > {output}'

rule bdg2npz:
    input: 'result/{eid}-{mark}.sorted.bedGraph'
    output: 'result/{eid}-{mark}.npz'
    conda:
        'environment.yaml'
    shell:
        'python bdg2npz.py '
        '-i {input} '
        '-c {hg19_sizes} '
        '-o {output}'
        
