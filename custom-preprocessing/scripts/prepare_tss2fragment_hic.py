import argparse
import pickle

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensg2tss', help='Input ensg2tss file.', required=True)
    parser.add_argument('--fragment-size', type=int, required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()

def load_pickle(f):
    with open(f, 'rb') as inFile:
        return pickle.load(inFile)

if __name__ == '__main__':
    args = parse_argument()

    ensg2tss = load_pickle(args.ensg2tss)
    fragment_size = args.fragment_size
    
    tss2fragment = {}
    for gene in ensg2tss.keys():
        for (chrom, start, end, strand) in ensg2tss[gene]:
            idx = start // fragment_size
            frag_start, frag_end = idx * fragment_size, (idx + 1) * fragment_size
            tss2fragment[f'{chrom}:{start}-{end}'] = f'{chrom}:{frag_start}-{frag_end}'
    
    with open(args.output, 'wb') as outFile:
        pickle.dump(tss2fragment, outFile)
    print(f'Saved tss2fragment file to {args.output}.')