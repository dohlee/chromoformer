import argparse
import gtfparse
import pickle

from collections import defaultdict

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='Input GTF file.', required=True)
    parser.add_argument('-o', '--output', help='Output ensg2tss pickle file.', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_argument()

    print(f'Loading input gtf file: {args.input}')
    gtf = gtfparse.read_gtf(args.input, expand_attribute_column=True)
    print('Done!')

    if any(gtf.gene_id.str.startswith('ENS')):
        gtf['gene_id'] = gtf.gene_id.str.split('.', expand=True)[0]

    gtf = gtf[(gtf.gene_type == 'protein_coding') & (gtf.feature == 'transcript')]
    gtf['transcript_length'] = gtf.end - gtf.start
    gtf = gtf.sort_values('transcript_length', ascending=False)

    ensg2tss = defaultdict(list)

    for transcript in gtf.to_records():
        if transcript.strand == '+':
            cses = (transcript.seqname, transcript.start - 1, transcript.start, transcript.strand)
        else:
            cses = (transcript.seqname, transcript.end - 1, transcript.end, transcript.strand)
        
        ensg2tss[transcript.gene_id].append(cses)
    
    with open(args.output, 'wb') as outFile:
        pickle.dump(ensg2tss, outFile)
    print(f'Saved ensg2tss file to {args.output}')