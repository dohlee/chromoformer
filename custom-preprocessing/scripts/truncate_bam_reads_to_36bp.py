import argparse
import pysam

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()

def truncate_cigar(cigars):
    c = 0
    new_cigars = []
    for flag, n in cigars:
        if flag == 2 or flag == 5:
            new_cigars.append((flag, n))
            continue

        if c + n > 36:
            l = 36 - c
            new_cigars.append((flag, l))
        else:
            new_cigars.append((flag, n))

        c += n
        if c > 36:
            break
    
    return new_cigars

if __name__ == '__main__':
    args = parse_arguments()

    bam = pysam.AlignmentFile(args.input)
    bam_out = pysam.AlignmentFile(args.output, 'wb', template=bam)

    for i, read in enumerate(bam):
        new_read = pysam.AlignedSegment()

        new_read.query_name = read.query_name
        new_read.query_sequence = read.seq[:36]
        new_read.flag = read.flag
        new_read.reference_id = read.reference_id
        new_read.reference_start = read.reference_start
        new_read.mapping_quality = read.mapping_quality
        new_read.cigar = truncate_cigar(read.cigar)
        new_read.next_reference_id = read.next_reference_id
        new_read.next_reference_start = read.next_reference_start
        new_read.template_length = read.template_length
        new_read.query_qualities = read.query_qualities[:36]
        # new_read.tags = read.tags

        bam_out.write(new_read)
        
        if i % 100000 == 0:
            print(f'Processed {i} reads.')

    bam.close()
    bam_out.close()
