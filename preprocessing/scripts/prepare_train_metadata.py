import argparse
import pandas as pd
import pickle
import time

from collections import defaultdict

def read_pickle(f):
    with open(f, 'rb') as inFile:
        return pickle.load(inFile)

def tuple2interval(t):
    return f'{t[0]}:{t[1]}-{t[2]}'

def split_interval(iv):
    chrom = iv.split(':')[0]
    start, end = map(int, iv.split(':')[1].split('-'))
    return chrom, start, end

def get_neighbors(ensg, ensg2tss, tss2fragment, frag2neighbors, pair2score, min_score=1.5, k=8, w_max=40000):
    tss_fragments = set(tss2fragment[tuple2interval(tss)] for tss in ensg2tss[ensg])

    neighbors = set()
    for tss_frag in tss_fragments:
        neighbor_frags = frag2neighbors[tss_frag]

        for neighbor_frag in neighbor_frags:
            neighbor_chrom, neighbor_start, neighbor_end = split_interval(neighbor_frag)
            score = pair2score[(tss_frag, neighbor_frag)]

            if score > min_score and end - start <= w_max:
                neighbors.add((neighbor_frag, score))

    neighbors = list(sorted(list(neighbors), key=lambda x: -x[1]))[:k]
    return neighbors

chromosome_splits = {
    1: ['chr1', 'chr6', 'chr5', 'chr8', 'chr14', 'chrY'],
    2: ['chr7', 'chr10', 'chr11', 'chr12', 'chr15', 'chr21'],
    3: ['chr2', 'chr3', 'chr4', 'chr16', 'chr18', 'chr20'],
    4: ['chr9', 'chr13', 'chr17', 'chr19', 'chr22', 'chrX'],
}

raw_exp = pd.read_csv('exp/raw_exp.tsv', sep='\t', index_col=0)
refgene = pd.read_csv('annotations/refGene.tsv', sep='\t')
tissue_mapping = pd.read_csv('annotations/tissue_mapping.csv')
eid2tissue = {r.eid:r.tissue for r in tissue_mapping.to_records()}
ensg2tss = read_pickle('annotations/ensg2tss.pickle')
tss2fragment = read_pickle('annotations/tss2fragment.pickle')

target_eids = tissue_mapping.eid.values
target_eids = ['E003']
target_ensgs = set(raw_exp.index.values) & set(refgene.ensg.values)

print(f'[prepare_train_metadata] There are {len(target_eids)} target EIDs and {len(target_ensgs)} target genes.')

exp = raw_exp.loc[target_ensgs][target_eids].sort_index()
exp.to_csv('exp/exp.csv')

# Prepare 0/1 labels.
eid2median = {}
for eid in target_eids:
    eid2median[eid] = exp[eid].median()

train = defaultdict(list)
for eid in target_eids:
    s = time.time()

    tissue = eid2tissue[eid]
    median = eid2median[eid]
    print(f'[prepare_train_metadata] Processing {eid} ({tissue}), median RPKM {median}.')

    frag2neighbors = read_pickle(f'annotations/{tissue}_frag2neighbors.pickle')
    pair2score = read_pickle(f'annotations/{tissue}_pair2score.pickle')

    for r in exp.reset_index()[['gene_id', eid]].to_records():
        train['gene_id'].append(r.gene_id)
        train['expression'].append(r[eid])
        train['eid'].append(eid)

        if r[eid] >= median:
            train['label'].append(1)
        else:
            train['label'].append(0)
         
        chrom, start, end, strand = ensg2tss[r.gene_id][0]
        train['chrom'].append(chrom)
        train['start'].append(start)
        train['end'].append(end)
        train['strand'].append(strand)
        
        # Assign CV split group for each gene according to the chromosome.
        for i in range(1, 5):
            if chrom in chromosome_splits[i]:
                train['split'].append(i)
                break

        # Add neighbors and score info.
        neighbor_tuples = get_neighbors(r.gene_id, ensg2tss, tss2fragment, frag2neighbors, pair2score, k=8)
        if len(neighbor_tuples) > 0:
            train['neighbors'].append(';'.join([x[0] for x in neighbor_tuples]))
            train['scores'].append(';'.join([str(x[1]) for x in neighbor_tuples]))
        else:
            train['neighbors'].append(None)
            train['scores'].append(None)

    print(f'[prepare_train_metadata] Done in {time.time() - s:.1f}s.')

train = pd.DataFrame(train)
train.to_csv('train.csv', index=False)

print('Done! Saved trained metadata to train.csv')
