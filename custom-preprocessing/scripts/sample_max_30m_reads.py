import argparse
import pysam
import random

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    bam = pysam.AlignmentFile(args.input)
    bam_out = pysam.AlignmentFile(args.output, 'wb', template=bam)

    reads = list(bam)

    # Sample reads.
    if len(reads) > 30000000:
        reads = random.sample(reads, 30000000)

    for read in reads:
        bam_out.write(read)

    bam_out.close()
