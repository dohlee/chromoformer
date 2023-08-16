import argparse
import torch
import random
import os

import numpy as np
import pandas as pd

from chromoformer import ChromoformerDataset
from chromoformer import ChromoformerClassifier
from tqdm import tqdm
from sklearn import metrics


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# Argument parsing.
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--meta", help="Path to input metadata file.", required=True)
parser.add_argument(
    "-d",
    "--npy-dir",
    help="Path to directory containing histone signals in .npy files.",
    required=True,
)
parser.add_argument("-o", "--output", help="Path to output expression prediction.", required=True)
parser.add_argument(
    "-w", "--weights", help="Path to pretrained Chromoformer weights in .pt format.", default=None
)
args = parser.parse_args()

#
# Parameter definitions.
#
seed = 123
bsz = 32
i_max = 8
w_prom = 40000
w_max = 40000
n_feats = 7
d_emb = 128
embed_kws = {
    "n_layers": 1,
    "n_heads": 2,
    "d_model": 128,
    "d_ff": 128,
}
pairwise_interaction_kws = {
    "n_layers": 2,
    "n_heads": 2,
    "d_model": 128,
    "d_ff": 256,
}
regulation_kws = {
    "n_layers": 6,
    "n_heads": 8,
    "d_model": 256,
    "d_ff": 256,
}
d_head = 128

seed_everything(seed)

meta = pd.read_csv(args.meta)
genes = meta.gene_id.tolist()
n_genes = len(genes)

print(f"Predicting expressions for {n_genes} genes.")

test_dataset = ChromoformerDataset(
    args.meta, args.npy_dir, genes, n_feats=7, i_max=i_max, w_prom=w_prom, w_max=w_max
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=bsz, num_workers=8, shuffle=False, drop_last=False
)

# Load pretrained weights.
model = ChromoformerClassifier(
    n_feats, d_emb, d_head, embed_kws, pairwise_interaction_kws, regulation_kws, seed=seed
)
if args.weights is not None:
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["net"])
model.cuda()

bar = tqdm(enumerate(test_loader, 1), total=len(test_loader))
predictions = []

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

        predictions.append(torch.sigmoid(out.cpu()).numpy()[:, 1])

predictions = np.concatenate(predictions)

# Write table annotated with Chromoformer prediction.
meta["prediction"] = predictions
meta.to_csv(args.output, index=False)

# Report.
auc = metrics.roc_auc_score(meta["label"], meta["prediction"])
ap = metrics.average_precision_score(meta["label"], meta["prediction"])
acc = metrics.accuracy_score(meta["label"], (meta["prediction"] > 0.5).astype(int))

print(f"ROC-AUC : {auc}")
print(f"Average Precision : {ap}")
print(f"Accuracy : {acc}")
