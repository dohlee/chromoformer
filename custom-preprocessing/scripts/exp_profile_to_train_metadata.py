import argparse
import pandas as pd

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input expression matrix file.', required=True)
    parser.add_argument('--target-genes', help='List of target gene IDs to train.', required=True)
    parser.add_argument('--id-col', default='gene_id')
    parser.add_argument('--exp-col', default='FPKM')
    parser.add_argument('--csv', action='store_true', default=False)
    parser.add_argument('-o', '--output', help='Output train metadata file.', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_argument()

    id_col, exp_col = args.id_col, args.exp_col

    if args.csv:
        exp = pd.read_csv(args.input)
    else:
        exp = pd.read_csv(args.input, sep='\t')

    if any(exp[id_col].str.startswith('ENS')):
        exp[id_col] = exp[id_col].str.split('.', expand=True)[0]

    if args.target_genes is not None:
        target_genes = list(open(args.target_genes, 'w').readlines())
        exp = exp[exp.gene_id.isin(target_genes)].groupby(id_col).agg({exp_col: 'mean'}).reset_index().rename({exp_col: 'expression'}, axis=1)
    else:
        exp = exp.groupby(id_col).agg({exp_col: 'mean'}).reset_index().rename({exp_col: 'expression'}, axis=1)

    exp['label'] = (exp.expression > exp.expression.median()).astype(int)
    exp[[id_col, 'expression', 'label']].to_csv(args.output, index=False)

