import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='Input bedgraph.')
parser.add_argument('-c', '--chrom-sizes', required=True, help='Chromosome size file.')
parser.add_argument('-o', '--output', required=True, help='Output numpy array.')
args = parser.parse_args()

chrom_sizes = pd.read_csv(args.chrom_sizes, names=['chrom', 'size'], sep='\t')
chrom2size = {r['chrom']:r['size'] for r in chrom_sizes.to_records()}

bdg = pd.read_csv(args.input, sep='\t', names=['chrom', 'start', 'end', 'depth'])

np_data = {}
for chrom in chrom2size.keys():
    np_data[chrom] = np.zeros(chrom2size[chrom], dtype=np.float16)

for interval in bdg.to_records():
    np_data[interval.chrom][interval.start:interval.end] = interval.depth

np.savez_compressed(args.output, **np_data)
