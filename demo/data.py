import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import random
import pickle

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter, defaultdict
from queue import PriorityQueue

def load_pickle(f):
    with open(f, 'rb') as inFile:
        return pickle.load(inFile)

class Roadmap3D(Dataset):
    
    def __init__(self, meta, eid, target_genes, i_max=8, w_prom=40000, w_max=40000):
        super(Roadmap3D, self).__init__()

        self.eid = eid

        self.target_genes = target_genes # List of ENSGs.

        meta = pd.read_csv(meta)
        self.meta = meta[meta.eid == self.eid].reset_index(drop=True)
        self.ensg2label = {r.gene_id:r.label for r in self.meta.to_records()}
        
        self.ensg2tss, self.ensg2pcres, self.ensg2scores = {}, {}, {}
        for r in self.meta.to_records():
            self.ensg2tss[r.gene_id] = (r.chrom, r.start, r.end, r.strand) 
            if not pd.isnull(r.neighbors):
                self.ensg2pcres[r.gene_id] = r.neighbors.split(';')
                self.ensg2scores[r.gene_id] = [float(s) for s in r.scores.split(';')]
            else:
                self.ensg2pcres[r.gene_id] = []
                self.ensg2scores[r.gene_id] = []

        self.i_max = i_max  # Maximum number of cis-interacting pCREs.
        self.w_prom = w_prom  # Promoter window size.
        self.w_max = w_max  # Maximum size of pCRE to consider.
    
    def _bin_and_pad(self, x, bin_size, max_n_bins):
        """Given a 2D tensor x, make binned tensor by taking average values of `bin_size` consecutive values.
        Appropriately pad by 
        left_pad = ceil((max_n_bins - n_bins) / 2)
        right_pad = floor((max_n_bins - n_bins) / 2)
        """
        l = x.size(1)
        n_bins = math.ceil(l / bin_size)

        # Binning.
        x_binned = []
        for i in range(n_bins):
            b = x[:, i * bin_size: (i + 1) * bin_size].mean(axis=1, keepdims=True)
            b = torch.log(b + 1)
            x_binned.append(b)
        x_binned = torch.cat(x_binned, axis=1)

        # Padding.
        left_pad = math.ceil( (max_n_bins - n_bins) / 2 )
        right_pad = math.floor( (max_n_bins - n_bins) / 2 )

        x_binned = torch.cat([
            torch.zeros([x.size(0), left_pad]),
            x_binned,
            torch.zeros([x.size(0), right_pad]),
        ], dim=1)

        return x_binned, left_pad, n_bins, right_pad
    
    def _get_region_representation(self, chrom, start, end, bin_size, max_n_bins, strand='+', window=None):
        x = torch.tensor(np.load(f'demo_data/{self.eid}/{chrom}:{start}-{end}.npy')).float()

        if window is not None:
            x = x[:, 20000 - window // 2:20000 + window // 2]
        x, left_pad, n_bins, right_pad = self._bin_and_pad(x, bin_size, max_n_bins)

        if strand == '+':
            return x, left_pad, n_bins, right_pad
        else:
            return torch.fliplr(x), right_pad, n_bins, left_pad

    def _split_interval_string(self, interval_string):
        """chrom:start-end -> (chrom, start, end)
        """
        chrom, tmp = interval_string.split(':')
        start, end = map(int, tmp.split('-'))

        return chrom, start, end

    def __getitem__(self, i):
        target_gene = self.target_genes[i]

        if target_gene not in self.ensg2tss:
            print(target_gene)

        pcres, scores = self.ensg2pcres[target_gene], self.ensg2scores[target_gene]
        n_partners, n_dummies = len(pcres), self.i_max - len(pcres)

        item = {}
        item['label'] = self.ensg2label[target_gene]

        chrom_p, start_p, end_p, strand_p = self.ensg2tss[target_gene]
        start_p, end_p = start_p - 20000, start_p + 20000

        for bin_size, max_n_bins in [(2000, self.w_max // 2000), (500, self.w_max // 500), (100, self.w_max // 100)]:
            x_pcres, mask_pcres = [], []

            x_p, left_pad_p, n_bins_p, right_pad_p = self._get_region_representation(chrom_p, start_p, end_p, bin_size, max_n_bins, strand_p, window=self.w_prom)
            x_p = x_p.permute(1, 0).unsqueeze(0) # 1 x max_n_bins x 7

            mask_p = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
            mask_p[0, left_pad_p:left_pad_p + n_bins_p, left_pad_p:left_pad_p + n_bins_p] = 0
            mask_p.unsqueeze_(0)

            interaction_freq = torch.zeros([self.i_max + 1, self.i_max + 1])

            for i, (score, pcre) in enumerate(zip(scores, pcres)):

                chrom_pcre, start_pcre, end_pcre = self._split_interval_string(pcre)
                if end_pcre - start_pcre > 40000:
                    print(target_gene, chrom_pcre, start_pcre, end_pcre)

                x_pcre, left_pad_pcre, n_bins_pcre, right_pad_pcre = self._get_region_representation(chrom_pcre, start_pcre, end_pcre, bin_size, max_n_bins)

                mask_pcre = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
                mask_pcre[0, left_pad_p:left_pad_p + n_bins_p, left_pad_pcre:left_pad_pcre + n_bins_pcre] = 0

                x_pcres.append(x_pcre)
                mask_pcres.append(mask_pcre)

                interaction_freq[0, i + 1] = score

            x_pcres.append(torch.zeros([7, n_dummies * max_n_bins]))
            x_pcres = torch.cat(x_pcres, axis=1).view(-1, self.i_max, max_n_bins)
            x_pcres = x_pcres.permute(1, 2, 0) # i_max x max_n_bins x 7

            for _ in range(n_dummies):
                m = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
                mask_pcres.append(m)

            interaction_mask = torch.ones([self.i_max + 1, self.i_max + 1], dtype=torch.bool)
            interaction_mask[:n_partners + 1, :n_partners + 1] = 0

            item[f'x_p_{bin_size}'] = x_p # 1 x max_n_bins x 7
            item[f'x_pcre_{bin_size}'] = x_pcres # i_max x max_n_bins x 7
            item[f'pad_mask_p_{bin_size}'] = mask_p # 1 x max_n_bins x max_n_bins
            item[f'pad_mask_pcre_{bin_size}'] = torch.stack(mask_pcres) # i_max x max_n_bins x max_n_bins
            item[f'interaction_mask_{bin_size}'] = interaction_mask.unsqueeze(0)
        
        item[f'interaction_freq'] = interaction_freq

        return item

    def __len__(self):
        return len(self.target_genes)