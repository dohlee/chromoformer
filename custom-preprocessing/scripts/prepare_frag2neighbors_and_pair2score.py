import argparse
import pickle
import numpy as np

from collections import defaultdict

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq-matrices', nargs='+', required=True)
    parser.add_argument('--tss2fragment', required=True)
    parser.add_argument('-c', '--chromosomes', nargs='+', required=True)
    parser.add_argument('--fragment-size', type=int, required=True)
    parser.add_argument('--freq-threshold', type=float, required=True)
    parser.add_argument('--output-frag2neighbors', required=True)
    parser.add_argument('--output-pair2score', required=True)
    return parser.parse_args()

def load_pickle(f):
    with open(f, 'rb') as inFile:
        return pickle.load(inFile)

if __name__ == '__main__':
    args = parse_argument()

    freq_fps = args.freq_matrices
    chromosomes = args.chromosomes
    fragment_size = args.fragment_size
    freq_threshold = args.freq_threshold

    assert len(freq_fps) == len(chromosomes)

    tss2fragment = load_pickle(args.tss2fragment)
    frag2neighbors = defaultdict(set)
    pair2score = dict()

    for chrom, freq_fp in zip(chromosomes, freq_fps):
        print(f'Processing chromosome {chrom}...')

        freq = np.loadtxt(freq_fp)
        print(f'\tLoaded normalized frequency matrix from {freq_fp}.')

        for tss, fragment in tss2fragment.items():

            if tss.split(':')[0] != chrom:
                continue

            fragment_idx = int(fragment.split(':')[1].split('-')[0]) // fragment_size
            neighbor_idxs = np.where(freq[fragment_idx] > args.freq_threshold)[0]

            for neighbor_idx in neighbor_idxs:
                neighbor_start, neighbor_end = neighbor_idx * fragment_size, (neighbor_idx + 1) * fragment_size
                neighbor = f'{chrom}:{neighbor_start}-{neighbor_end}'

                frag2neighbors[fragment].add(neighbor)
                pair2score[(fragment, neighbor)] = freq[fragment_idx, neighbor_idx]
    
    with open(args.output_frag2neighbors, 'wb') as outFile:
        pickle.dump(frag2neighbors, outFile)
    print(f'Saved frag2neighbors file to {args.output_frag2neighbors}. Total {len(frag2neighbors)} fragment-neighbor pairs are saved.')

    with open(args.output_pair2score, 'wb') as outFile:
        pickle.dump(pair2score, outFile)
    print(f'Saved pair2score file to {args.output_pair2score}. Scores for total {len(pair2score)} pairs are saved.')