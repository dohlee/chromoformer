import argparse
import pickle
import numpy as np
import pandas as pd
import os

from collections import defaultdict
from queue import PriorityQueue
from pathlib import Path

MAX_DIST = 40000
MAX_PARTNERS = 8

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ensg2tss', required=True)
    parser.add_argument('--tss2fragment', required=True)
    parser.add_argument('--frag2neighbors', required=True)
    parser.add_argument('--pair2score', required=True)
    parser.add_argument('--npz-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--freq-threshold', type=float, required=True)
    parser.add_argument('-m', '--marks', nargs='+', default=['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K36me3', 'H3K27ac', 'H3K9ac'])
    parser.add_argument('-c', '--chromosomes', nargs='+', required=True)
    return parser.parse_args()

def split_interval_string(s):
    c = s.split(':')[0]
    s, e = map(int, s.split(':')[1].split('-'))
    return c, s, e

def load_pickle(f):
    with open(f, 'rb') as inFile:
        return pickle.load(inFile)

def get_partners(ensg, ensg2tss, tss2fragment, frag2neighbors, pair2score, freq_threshold):
    partner_pool = PriorityQueue()
    visited_partner = set()

    n_partners = 0
    tss_list = ensg2tss[ensg]
    for c, s, e, st in tss_list:
        tss = f'{c}:{s}-{e}'
        tss_frag = tss2fragment[tss]

        _partner_pool = [(pair2score[(tss_frag, partner)], partner) for partner in frag2neighbors[tss_frag]]
        for freq, partner in _partner_pool:
            if partner not in visited_partner and freq > freq_threshold:
                _c, _s, _e = split_interval_string(partner)
                if _e - _s > MAX_DIST:
                    continue
                
                partner_pool.put((-freq, partner))
                visited_partner.add(partner)
                n_partners += 1
        
    n_partners = min(n_partners, MAX_PARTNERS)
    partners = [partner_pool.get() for _ in range(n_partners)]

    return [partner for _, partner in partners]

def get_fragment_universe(ensg2tss, tss2fragment, frag2neighbors, pair2score, freq_threshold):
    fragment_universe = set()

    for ensg, tss in ensg2tss.items():
        fragment_universe |= set(get_partners(ensg, ensg2tss, tss2fragment, frag2neighbors, pair2score, freq_threshold))

    return fragment_universe

if __name__ == '__main__':
    args = parse_argument()

    chromosomes = args.chromosomes
    marks = args.marks
    ensg2tss = load_pickle(args.ensg2tss) 
    tss2fragment = load_pickle(args.tss2fragment)
    frag2neighbors = load_pickle(args.frag2neighbors)
    pair2score = load_pickle(args.pair2score)

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)

    print('Gathering fragment universe...')
    fragment_universe = get_fragment_universe(ensg2tss, tss2fragment, frag2neighbors, pair2score, args.freq_threshold)
    print(f'Done! Total {len(fragment_universe)} fragments will be processed.')

    print('Loading npz signals...')
    signal = defaultdict(dict)
    for mark in marks:
        npy = np.load(npz_dir / f'{mark}.npz')

        for chrom in chromosomes:
            signal[mark][chrom] = npy[chrom]
    print('Done!')

    print('Saving fragment npy files...')
    for region in fragment_universe:
        if os.path.exists(npz_dir / f'{region}.npy'):
            continue
        
        if pd.isnull(region):
            continue

        chrom = region.split(':')[0]
        if chrom not in chromosomes:
            continue
        
        start, end = map(int, region.split(':')[1].split('-'))
        
        v = []
        for mark in args.marks:
            v.append(signal[mark][chrom][start:end])
        v = np.array(v)

        np.save(out_dir / f'{region}.npy')
    print('Done!')