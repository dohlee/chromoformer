import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
import wandb
import yaml

from tqdm import tqdm
from scipy import stats
from sklearn import metrics

from .data import ChromoformerDataset
from .net import ChromoformerClassifier, ChromoformerRegressor
from .util import seed_everything


torch.autograd.set_detect_anomaly(True)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-c", "--config", required=True)
parser.add_argument("--exp-id", required=True)
parser.add_argument("-m", "--meta", required=True)
parser.add_argument("-d", "--npy-dir", required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--binsizes", nargs="+", default=[2000, 500, 100])
parser.add_argument("--regression", action="store_true", default=False)
parser.add_argument("--use-wandb", action="store_true", default=False)

args = parser.parse_args()

#
# Training setup.
#
if args.use_wandb is False:
    os.environ["WANDB_MODE"] = "disabled"

with open(args.config) as f:
    config = yaml.safe_load(f)

print(config)

config["exp_id"] = args.exp_id  # Override
exp_id = args.exp_id
seed = config["seed"]
num_epoch = config["num_epoch"]
lr = config["lr"]
bsz = config["bsz"]
gamma = config["gamma"]

i_max = config["i_max"]
w_prom = config["w_prom"]
w_max = config["w_max"]

n_feats = config["n_feats"]
d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
pairwise_interaction_kws = config["pairwise_interaction"]
regulation_kws = config["regulation"]

d_head = config["d_head"]

#
# Setup end.
#

seed_everything(seed)
wandb.init(project="chromoformer-refactoring", entity="dohlee", group=exp_id)
wandb.config.update(args)
wandb.config.update(config)

meta = (
    pd.read_csv(args.meta).sample(frac=1, random_state=seed).reset_index(drop=True)
)  # load and shuffle.

if args.regression and "expression" not in meta.columns:
    raise ValueError(
        "`expression` column is required for training ChromoformerRegression model."
    )

# Split genes into two sets (train/val).
genes = set(meta.gene_id.unique())
n_genes = len(genes)
print("Target genes:", len(genes))

qs = [
    meta[meta.split == 1].gene_id.tolist(),
    meta[meta.split == 2].gene_id.tolist(),
    meta[meta.split == 3].gene_id.tolist(),
    meta[meta.split == 4].gene_id.tolist(),
]

train_genes = (
    qs[(args.fold + 0) % 4] + qs[(args.fold + 1) % 4] + qs[(args.fold + 2) % 4]
)
val_genes = qs[(args.fold + 3) % 4]

wandb.config.update(
    {
        "n_train_genes": len(train_genes),
        "n_val_genes": len(val_genes),
    }
)

print(len(train_genes), len(val_genes))

train_dataset = ChromoformerDataset(
    args.meta,
    args.npy_dir,
    train_genes,
    n_feats,
    i_max,
    args.binsizes,
    w_prom,
    w_max,
    regression=args.regression,
)
val_dataset = ChromoformerDataset(
    args.meta,
    args.npy_dir,
    val_genes,
    n_feats,
    i_max,
    args.binsizes,
    w_prom,
    w_max,
    regression=args.regression,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bsz, num_workers=8, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, num_workers=8)

Model = ChromoformerRegressor if args.regression else ChromoformerClassifier
model = Model(
    n_feats,
    d_emb,
    d_head,
    embed_kws,
    pairwise_interaction_kws,
    regulation_kws,
    binsizes=args.binsizes,
    seed=42,
)

model.cuda()

criterion = nn.MSELoss() if args.regression else nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

optimizer.zero_grad()
optimizer.step()

for epoch in range(1, num_epoch):
    # Prepare train.
    bar = tqdm(enumerate(train_loader, 1), total=len(train_loader))
    running_loss = 0.0
    train_out, train_label = [], []

    # Train.
    model.train()
    for batch, d in bar:
        for k, v in d.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    v[_k] = _v.cuda()
            else:
                d[k] = v.cuda()

        if args.regression:
            d["label"] = d["label"].view(-1, 1)

        optimizer.zero_grad()

        out = model(
            d["promoter_feats"],
            d["promoter_pad_masks"],
            d["pcre_feats"],
            d["pcre_pad_masks"],
            d["interaction_masks"],
            d["interaction_freq"],
        )

        loss = criterion(out, d["label"])

        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        running_loss += loss

        train_out.append(out.detach().cpu())
        train_label.append(d["label"].cpu())

        if batch % 10 == 0:
            batch_loss = running_loss / 10.0

            train_out, train_label = map(torch.cat, (train_out, train_label))

            if args.regression:
                train_pred = train_out.flatten()
                train_label = train_label.flatten()

                batch_r2 = metrics.r2_score(train_label, train_pred) * 100
                batch_r = stats.pearsonr(train_label, train_pred)[0] * 100

                bar.set_description(
                    f"E{epoch} {batch_loss:.4f}, lr={get_lr(optimizer)}, r2={batch_r2:.4f}, r={batch_r:.4f}"
                )

                wandb.log(
                    {
                        "train/loss": batch_loss,
                        "train/r2": batch_r2,
                        "train/r": batch_r,
                    }
                )
            else:
                train_score = train_out.softmax(axis=1)[:, 1]
                train_pred = train_out.argmax(axis=1)

                batch_acc = metrics.accuracy_score(train_label, train_pred) * 100
                batch_auc = metrics.roc_auc_score(train_label, train_score) * 100
                batch_ap = (
                    metrics.average_precision_score(train_label, train_score) * 100
                )

                bar.set_description(
                    f"E{epoch} {batch_loss:.4f}, lr={get_lr(optimizer)}, acc={batch_acc:.4f}, auc={batch_auc:.4f}, ap={batch_ap:.4f}"
                )

                wandb.log(
                    {
                        "train/loss": batch_loss,
                        "train/acc": batch_acc,
                        "train/auc": batch_auc,
                        "train/ap": batch_ap,
                    }
                )

            running_loss = 0.0
            train_out, train_label = [], []

    # Prepare validation.
    bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
    val_out, val_label = [], []

    # Validation.
    model.eval()
    with torch.no_grad():
        for batch, d in bar:
            for k, v in d.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        v[_k] = _v.cuda()
                else:
                    d[k] = v.cuda()

            out = model(
                d["promoter_feats"],
                d["promoter_pad_masks"],
                d["pcre_feats"],
                d["pcre_pad_masks"],
                d["interaction_masks"],
                d["interaction_freq"],
            )
            val_out.append(out.cpu())

            val_label.append(d["label"].cpu())

    val_out = torch.cat(val_out)
    val_label = torch.cat(val_label)

    val_loss = criterion(val_out, val_label)

    # Metrics.
    val_label = val_label.numpy()
    if args.regression:
        val_score = val_out.flatten().numpy()

        val_r2 = metrics.r2_score(val_label, val_score) * 100
        val_r = stats.pearsonr(val_label, val_score)[0] * 100

        print(f"Validation loss={val_loss:.4f}, r2={val_r2:.4f}, r={val_r:.4f}")
        wandb.log(
            {
                "val/loss": val_loss,
                "val/r2": val_r2,
                "val/r": val_r,
            }
        )
    else:
        val_score = val_out.softmax(axis=1)[:, 1].numpy()
        val_pred = val_out.argmax(axis=1).numpy()

        val_acc = metrics.accuracy_score(val_label, val_pred) * 100
        val_auc = metrics.roc_auc_score(val_label, val_score) * 100
        val_ap = metrics.average_precision_score(val_label, val_score) * 100

        print(
            f"Validation loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}, ap={val_ap:.4f}"
        )
        wandb.log(
            {
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/auc": val_auc,
                "val/ap": val_ap,
                "val/epoch": epoch,
            }
        )

    if args.regression:
        ckpt = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "last_val_loss": val_loss,
            "last_val_r2": val_r2,
            "val_score": val_score,
            "val_label": val_label,
        }
        torch.save(ckpt, args.output)
    else:
        ckpt = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "last_val_loss": val_loss,
            "last_val_auc": val_auc,
            "val_score": val_score,
            "val_label": val_label,
        }
        torch.save(ckpt, args.output)
    scheduler.step()

if args.regression:
    wandb.summary.update(
        {
            "last_val_loss": val_loss,
            "last_val_r2": val_r2,
        }
    )
else:
    wandb.summary.update(
        {
            "last_val_loss": val_loss,
            "last_val_auc": val_auc,
        }
    )
