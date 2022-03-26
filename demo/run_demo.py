import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import numpy as np
import pandas as pd
import yaml

from data import Roadmap3D
from net import Chromoformer
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from sklearn import metrics

from util import seed_everything

# Argument parsing.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Path to input metadata file.', required=True)
parser.add_argument('-o', '--output', help='Path to output expression prediction.', required=True)
parser.add_argument('-w', '--weights', help='Path to pretrained Chromoformer weights in .pt format.', default=None)
parser.add_argument('-e', '--eid', help='Specify cell type in EID.', required=True)
args = parser.parse_args()

#
# Parameter definitions.
#
seed = 123
num_epoch = 10
bsz = 32

i_max = 8
w_prom = 40000
w_max = 40000

n_feats = 7
embed_n_layers = 1
embed_n_heads = 2
embed_d_model = 128
embed_d_ff = 128

pw_int_n_layers = 2
pw_int_n_heads = 2
pw_int_d_model = 128
pw_int_d_ff = 256

reg_n_layers = 6
reg_n_heads = 8
reg_d_model = 256
reg_d_ff = 256

head_n_feats = 128

seed_everything(seed)

meta = pd.read_csv(args.input)
meta = meta[meta.eid == args.eid]

genes = meta.gene_id.tolist()
n_genes = len(genes)

print(f'Predicting expressions for {n_genes} genes.')

test_dataset = Roadmap3D(args.input, args.eid, genes, i_max, w_prom, w_max)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bsz, num_workers=8, shuffle=False, drop_last=False)

# Load pretrained weights.
model = Chromoformer(
    n_feats, embed_n_layers, embed_n_heads, embed_d_model, embed_d_ff,
    pw_int_n_layers, pw_int_n_heads, pw_int_d_model, pw_int_d_ff,
    reg_n_layers, reg_n_heads, reg_d_model, reg_d_ff, head_n_feats,
)
if args.weights is not None:
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['net'])
model.cuda()

bar = tqdm(enumerate(test_loader, 1), total=len(test_loader))
predictions = []

model.eval()
with torch.no_grad():
    for batch, d in bar:
        for k, v in d.items():
            d[k] = v.cuda()
        
        out = model(
            d['x_p_2000'], d['pad_mask_p_2000'], d['x_pcre_2000'], d['pad_mask_pcre_2000'], d['interaction_mask_2000'],
            d['x_p_500'], d['pad_mask_p_500'], d['x_pcre_500'], d['pad_mask_pcre_500'], d['interaction_mask_2000'],
            d['x_p_100'], d['pad_mask_p_100'], d['x_pcre_100'], d['pad_mask_pcre_100'], d['interaction_mask_2000'],
            d['interaction_freq'],
        )

        predictions.append(torch.sigmoid(out.cpu()).numpy()[:, 1])

predictions = np.concatenate(predictions)

# Write table annotated with Chromoformer prediction.
meta['prediction'] = predictions
meta.to_csv(args.output, index=False)

# Report.
auc = metrics.roc_auc_score(meta["label"], meta["prediction"])
ap = metrics.average_precision_score(meta["label"], meta["prediction"])
acc = metrics.accuracy_score(meta["label"], (meta["prediction"] > 0.5).astype(int))

print(f'ROC-AUC : {auc}')
print(f'Average Precision : {ap}')
print(f'Accuracy : {acc}')