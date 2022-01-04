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

import wandb

train_dir = Path('../preprocessing')
train_meta = train_dir / 'train.csv'

torch.autograd.set_detect_anomaly(True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-c', '--config', required=True)
parser.add_argument('--exp-id', required=True)
parser.add_argument('--eid', required=True)
parser.add_argument('--fold', type=int, required=True)
parser.add_argument('--debug', action='store_true', default=False)

args = parser.parse_args()

#
# Training setup.
#
if args.debug:
    os.environ['WANDB_MODE'] = 'offline'

with open(args.config) as f:
    config = yaml.safe_load(f)

print(config)

exp_id = args.exp_id
seed = config['seed']
num_epoch = config['num_epoch']
lr = config['lr']
bsz = config['bsz']
gamma = config['gamma']

i_max = config['i_max']
w_prom = config['w_prom']
w_max = config['w_max']

n_feats = config['n_feats']
embed_n_layers = config['embed_n_layers']
embed_n_heads = config['embed_n_heads']
embed_d_model = config['embed_d_model']
embed_d_ff = config['embed_d_ff']

pw_int_n_layers = config['pw_int_n_layers']
pw_int_n_heads = config['pw_int_n_heads']
pw_int_d_model = config['pw_int_d_model']
pw_int_d_ff = config['pw_int_d_ff']

reg_n_layers = config['reg_n_layers']
reg_n_heads = config['reg_n_heads']
reg_d_model = config['reg_d_model']
reg_d_ff = config['reg_d_ff']

head_n_feats = config['head_n_feats']

group_id = f'{exp_id}-{args.eid}'
# 
# Setup end.
#

seed_everything(seed)
wandb.init(project='chromoformer-reproduction', entity='dohlee', group=group_id)
wandb.config.update(args)
wandb.config.update(config)

meta = pd.read_csv(train_meta) \
    .sample(frac=1, random_state=seed) \
    .reset_index(drop=True) # load and shuffle.

meta = meta[meta.eid == args.eid]

# Split genes into two sets (train/val).
genes = set(meta.gene_id.unique())
n_genes = len(genes)
print('Target genes:', len(genes))

qs = [
    meta[meta.split == 1].gene_id.tolist(),
    meta[meta.split == 2].gene_id.tolist(),
    meta[meta.split == 3].gene_id.tolist(),
    meta[meta.split == 4].gene_id.tolist(),
]

train_genes = qs[(args.fold + 0) % 4] + qs[(args.fold + 1) % 4] + qs[(args.fold + 2) % 4]
val_genes = qs[(args.fold + 3) % 4]

wandb.config.update({
    'n_train_genes': len(train_genes),
    'n_val_genes': len(val_genes),
})

print(len(train_genes), len(val_genes))

train_dataset = Roadmap3D(args.eid, train_genes, i_max, w_prom, w_max)
val_dataset = Roadmap3D(args.eid, val_genes, i_max, w_prom, w_max)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, num_workers=8, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, num_workers=8)

model = Chromoformer(
    n_feats, embed_n_layers, embed_n_heads, embed_d_model, embed_d_ff,
    pw_int_n_layers, pw_int_n_heads, pw_int_d_model, pw_int_d_ff,
    reg_n_layers, reg_n_heads, reg_d_model, reg_d_ff, head_n_feats,
)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

optimizer.zero_grad()
optimizer.step()

best_val_auc = 0
for epoch in range(1, num_epoch):

    # Prepare train.
    bar = tqdm(enumerate(train_loader, 1), total=len(train_loader))
    running_loss = 0.0
    train_out, train_label = [], []

    # Train.
    model.train()
    for batch, d in bar:
        for k, v in d.items():
            d[k] = v.cuda()

        optimizer.zero_grad()

        out = model(
            d['x_p_2000'], d['pad_mask_p_2000'], d['x_pcre_2000'], d['pad_mask_pcre_2000'], d['interaction_mask_2000'],
            d['x_p_500'], d['pad_mask_p_500'], d['x_pcre_500'], d['pad_mask_pcre_500'], d['interaction_mask_2000'],
            d['x_p_100'], d['pad_mask_p_100'], d['x_pcre_100'], d['pad_mask_pcre_100'], d['interaction_mask_2000'],
            d['interaction_freq'],
        )
        loss = criterion(out, d['label'])

        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        running_loss += loss

        train_out.append(out.detach().cpu())
        train_label.append(d['label'].cpu())

        if batch % 10 == 0:
            batch_loss = running_loss / 10.

            train_out, train_label = map(torch.cat, (train_out, train_label))
            train_score = train_out.softmax(axis=1)[:, 1]
            train_pred = train_out.argmax(axis=1)

            batch_acc = metrics.accuracy_score(train_label, train_pred) * 100
            batch_auc = metrics.roc_auc_score(train_label, train_score) * 100
            batch_ap = metrics.average_precision_score(train_label, train_score) * 100

            bar.set_description(f'E{epoch} {batch_loss:.4f}, lr={get_lr(optimizer)}, acc={batch_acc:.4f}, auc={batch_auc:.4f}, ap={batch_ap:.4f}')

            running_loss = 0.0
            train_out, train_label = [], []

            wandb.log({
                'train/loss': batch_loss,
                'train/acc': batch_acc,
                'train/auc': batch_auc,
                'train/ap': batch_ap,
            })

    # Prepare validation.
    bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
    val_out, val_label = [], []

    # Validation.
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
            val_out.append(out.cpu())

            val_label.append(d['label'].cpu())
    
    val_out = torch.cat(val_out)
    val_label = torch.cat(val_label)

    val_loss = criterion(val_out, val_label)

    # Metrics.
    val_label = val_label.numpy()
    val_score = val_out.softmax(axis=1)[:, 1].numpy()
    val_pred = val_out.argmax(axis=1).numpy()

    val_acc = metrics.accuracy_score(val_label, val_pred) * 100
    val_auc = metrics.roc_auc_score(val_label, val_score) * 100
    val_ap = metrics.average_precision_score(val_label, val_score) * 100

    print(f'Validation loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}, ap={val_ap:.4f}')

    wandb.log({
        'val/loss': val_loss,
        'val/acc': val_acc,
        'val/auc': val_auc,
        'val/ap': val_ap,
        'val/epoch': epoch,
    })

    ckpt = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'last_val_loss': val_loss,
        'last_val_auc': val_auc,
        'val_score': val_score,
        'val_label': val_label,
    }
    torch.save(ckpt, args.output)
    scheduler.step()

wandb.summary.update({
    'last_val_loss': val_loss,
    'last_val_auc': val_auc,
})
            
