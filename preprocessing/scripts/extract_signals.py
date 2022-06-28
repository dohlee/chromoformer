import argparse
import os
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

marks = ['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K36me3', 'H3K27ac', 'H3K9ac']
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input train metadata file.')
    parser.add_argument('-e', '--eid', required=True, help='Reference epigenome ID to process.')
    parser.add_argument('-o', '--output', required=True, help='Output directory.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    eid = args.eid

    train = pd.read_csv(args.input)
    train = train[train.eid == eid].reset_index(drop=True)

    region_set = set()
    for r in train.to_records():
        # Add TSS to region set.
        region_set.add(f'{r.chrom}:{r.start - 20000}-{r.start + 20000}')

        # Add neighbors to region set.
        if not pd.isnull(r.neighbors):
            for n in r.neighbors.split(';'):
                region_set.add(n)

    print('Loading genomewide read depth signals...')
    s = time.time()
    signal = defaultdict(dict)
    for mark in marks:
        print(f'Reading {mark}.')
        npy = np.load(f'hist/{eid}-{mark}.npz')

        for chrom in chromosomes:
            signal[mark][chrom] = npy[chrom]
    print(f'Done in {time.time() - s:.1f}s.')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for region in tqdm(region_set):
        out = os.path.join(args.output, f'{region}.npy')
        if os.path.exists(out):
            continue

        if pd.isnull(region):
            continue

        chrom = region.split(':')[0]
        if chrom not in chromosomes:
            continue
        start, end = map(int, region.split(':')[1].split('-'))

        v = []
        for mark in marks:
            v.append(signal[mark][chrom][start:end])
        v = np.array(v)

        np.save(out, v)

