import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from modules import Transformer, PairwiseTransformer

class EmbeddingTransformer(nn.Module):
    def __init__(self, n_feats, n_layers, n_heads, d_model, d_ff, pos_enc=True, activation=F.relu):
        super(EmbeddingTransformer, self).__init__()

        self.d_model = d_model
        self.lin_proj = nn.Linear(n_feats, self.d_model, bias=False)

        self.transformer = Transformer(n_layers, n_heads, self.d_model, self.d_model, d_ff, activation, gate=False)
        self.pos_enc = pos_enc

    def _pos_enc(self, dim, max_len):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        k = torch.exp(-np.log(10000) * torch.arange(0, dim, 2) / dim)
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        return pe
    
    def forward(self, x, mask):
        """
        x : bsz x (max_n_interactions + 1) x max_n_bins x n_mark
        mask : bsz x (max_n_interactions + 1) x 1 x max_n_bins x max_n_bins
        """
        bsz, max_n_bins, n_mark = mask.size(0), mask.size(4), x.size(3)
        mask = mask.view(-1, 1, max_n_bins, max_n_bins)
        # --> mask : (bsz x (max_n_interactions + 1)) x 1 x max_n_bins x max_n_bins

        x = x.view(-1, max_n_bins, n_mark)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x n_maxk
        x = self.lin_proj(x)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x d_model

        # Embed and linearly project each genomic region independently.
        # Add positional embedding.
        if self.pos_enc:
            x = x + self._pos_enc(dim=self.d_model, max_len=max_n_bins)[:, :self.d_model].unsqueeze(0).cuda()
        # --> x : (bsz x (max_n_interactions + 1)) x max_n_bins x d_model
        x = self.transformer(x, mask)
        # --> x : (bsz x (max_n_interactions + 1)) x max_n_bins x d_model
        x = x.view(bsz, -1, max_n_bins, self.d_model)

        return x, x[:, :, max_n_bins // 2]

class PairwiseInteractionTransformer(nn.Module):
    def __init__(self, n_feats_p, n_feats_pcre, n_layers, n_heads, d_model, d_ff, pos_enc=True, res_p=0.0, att_p=0.0, activation=F.relu):
        super(PairwiseInteractionTransformer, self).__init__()

        self.d_model = d_model

        self.ln = nn.LayerNorm(self.d_model)
        self.lin_proj_p = nn.Linear(n_feats_p, self.d_model, bias=False)
        self.lin_proj_pcre = nn.Linear(n_feats_pcre, self.d_model, bias=False)

        self.transformer = PairwiseTransformer(n_layers, n_heads, self.d_model, self.d_model, d_ff, res_p, att_p, activation, gate=False)
        self.pos_enc = pos_enc

    def _pos_enc(self, dim, max_len):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        k = torch.exp(-np.log(10000) * torch.arange(0, dim, 2) / dim)
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        return pe
    
    def forward(self, x_p, x_pcre, mask):
        """
        x : bsz x (max_n_interactions + 1) x max_n_bins x n_mark
        mask : bsz x (max_n_interactions + 1) x 1 x max_n_bins x max_n_bins
        """
        bsz, max_n_bins, n_mark = mask.size(0), mask.size(4), x_pcre.size(3)
        mask = mask.view(-1, 1, max_n_bins, max_n_bins)
        # --> mask : (bsz x (max_n_interactions + 1)) x 1 x max_n_bins x max_n_bins

        x_p = x_p.expand(-1, x_pcre.size(1), -1, -1)
        x_p = x_p.reshape(-1, x_p.size(2), x_p.size(3))
        x_pcre = x_pcre.view(-1, max_n_bins, n_mark)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x n_maxk
        x_p = self.lin_proj_p(x_p)
        x_pcre = self.lin_proj_pcre(x_pcre)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x d_model

        # Embed and linearly project each genomic region independently.
        # Add positional embedding.
        if self.pos_enc:
            x_pcre = x_pcre + self._pos_enc(dim=self.d_model, max_len=max_n_bins)[:, :self.d_model].unsqueeze(0).cuda()
        # --> x : (bsz x (max_n_interactions + 1)) x max_n_bins x d_model
        x = self.transformer(x_p, x_pcre, mask)
        # --> x : (bsz x (max_n_interactions + 1)) x max_n_bins x d_model

        x = x.view(bsz, -1, max_n_bins, self.d_model)
        # --> x : bsz x (max_n_interactions + 1) x max_n_bins x d_model

        x = x[:, :, max_n_bins // 2]  # Take the genomic bin representation at center.
        return x

class RegulationTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_emb, d_model, d_ff, pos_enc=True, activation=F.relu):
        super(RegulationTransformer, self).__init__()

        self.transformer = Transformer(n_layers, n_heads, d_emb, d_model, d_ff, gate=True)
    
    def forward(self, x, mask, bias):
        return self.transformer(x, mask, bias)

class Chromoformer(nn.Module):
    def __init__(
        self, 
        n_feats=7, embed_n_layers=1, embed_n_heads=2, embed_d_model=128, embed_d_ff=128,
        pw_int_n_layers=2, pw_int_n_heads=2, pw_int_d_model=128, pw_int_d_ff=256,
        reg_n_layers=6, reg_n_heads=8, reg_d_model=256, reg_d_ff=256, head_n_feats=128,
    ):
        super(Chromoformer, self).__init__()
        self.embed2000 = EmbeddingTransformer(
            n_feats, embed_n_layers, embed_n_heads, embed_d_model, embed_d_ff
        )
        self.embed500 = EmbeddingTransformer(
            n_feats, embed_n_layers, embed_n_heads, embed_d_model, embed_d_ff
        )
        self.embed100 = EmbeddingTransformer(
            n_feats, embed_n_layers, embed_n_heads, embed_d_model, embed_d_ff
        )

        self.pw_int2000 = PairwiseInteractionTransformer(
            embed_d_model, n_feats, pw_int_n_layers, pw_int_n_heads, pw_int_d_model, pw_int_d_ff
        )
        self.pw_int500 = PairwiseInteractionTransformer(
            embed_d_model, n_feats, pw_int_n_layers, pw_int_n_heads, pw_int_d_model, pw_int_d_ff
        )
        self.pw_int100 = PairwiseInteractionTransformer(
            embed_d_model, n_feats, pw_int_n_layers, pw_int_n_heads, pw_int_d_model, pw_int_d_ff
        )

        self.reg2000 = RegulationTransformer(
            reg_n_layers, reg_n_heads, embed_d_model, reg_d_model, reg_d_ff
        )
        self.reg500 = RegulationTransformer(
            reg_n_layers, reg_n_heads, embed_d_model, reg_d_model, reg_d_ff
        )
        self.reg100 = RegulationTransformer(
            reg_n_layers, reg_n_heads, embed_d_model, reg_d_model, reg_d_ff
        )

        self.fc_head = nn.Sequential(
            nn.Linear(embed_d_model * 3, head_n_feats),
            nn.ReLU(),
            nn.Linear(head_n_feats, 2),
        )

    def forward(self,
        x_p_2000, pad_mask_p_2000, x_pcre_2000, pad_mask_pcre_2000, interaction_mask_2000,
        x_p_500, pad_mask_p_500, x_pcre_500, pad_mask_pcre_500, interaction_mask_500,
        x_p_100, pad_mask_p_100, x_pcre_100, pad_mask_pcre_100, interaction_mask_100,
        interaction_freq,
    ):
        p_emb_2000_total, p_emb_2000 = self.embed2000(x_p_2000, pad_mask_p_2000)
        p_emb_500_total, p_emb_500 = self.embed500(x_p_500, pad_mask_p_500)
        p_emb_100_total, p_emb_100 = self.embed100(x_p_100, pad_mask_p_100)

        pw_int_emb_2000 = self.pw_int2000(p_emb_2000_total, x_pcre_2000, pad_mask_pcre_2000)
        pw_int_emb_500 = self.pw_int500(p_emb_500_total, x_pcre_500, pad_mask_pcre_500)
        pw_int_emb_100 = self.pw_int100(p_emb_100_total, x_pcre_100, pad_mask_pcre_100)

        x_2000 = torch.cat([p_emb_2000, pw_int_emb_2000], axis=1)
        x_500 = torch.cat([p_emb_500, pw_int_emb_500], axis=1)
        x_100 = torch.cat([p_emb_100, pw_int_emb_100], axis=1)

        emb = torch.cat([x_2000[:, 0], x_500[:, 0], x_100[:, 0]], axis=1)

        x_2000 = self.reg2000(x_2000, interaction_mask_2000, bias=interaction_freq)
        x_500 = self.reg500(x_500, interaction_mask_500, bias=interaction_freq)
        x_100 = self.reg100(x_100, interaction_mask_100, bias=interaction_freq)

        x = torch.cat([x_2000[:, 0], x_500[:, 0], x_100[:, 0]], axis=1) + emb # Take representation of the target promoter.
        return self.fc_head(x)

if __name__ == '__main__':
    # import data
    import tqdm
    import pandas as pd

    model = Chromoformer().cuda()

    # Dummy data.
    bsz = 8
    i_max = 8

    x_p_2000, x_p_500, x_p_100 = torch.randn([bsz, 1, 20, 7]), torch.randn([bsz, 1, 80, 7]), torch.randn([bsz, 1, 400, 7])
    x_pcre_2000, x_pcre_500, x_pcre_100 = torch.randn([bsz, i_max, 20, 7]), torch.randn([bsz, i_max, 80, 7]), torch.randn([bsz, i_max, 400, 7])

    pad_mask_p_2000, pad_mask_p_500, pad_mask_p_100 = torch.randn([bsz, 1, 1, 20, 20]).bool(), torch.randn([bsz, 1, 1, 80, 80]).bool(), torch.randn([bsz, 1, 1, 400, 400]).bool()
    pad_mask_pcre_2000, pad_mask_pcre_500, pad_mask_pcre_100 = torch.randn([bsz, i_max, 1, 20, 20]).bool(), torch.randn([bsz, i_max, 1, 80, 80]).bool(), torch.randn([bsz, i_max, 1, 400, 400]).bool()

    interaction_mask_2000, interaction_mask_500, interaction_mask_100 = torch.randn([bsz, 1, 1 + i_max, 1 + i_max]).bool(), torch.randn([bsz, 1, 1 + i_max, 1 + i_max]).bool(), torch.randn([bsz, 1, 1 + i_max, 1 + i_max]).bool()
    interaction_freq = torch.randn([bsz, 1 + i_max, 1 + i_max])

    x_p_2000, x_p_500, x_p_100 = x_p_2000.cuda(), x_p_500.cuda(), x_p_100.cuda()
    x_pcre_2000, x_pcre_500, x_pcre_100 = x_pcre_2000.cuda(), x_pcre_500.cuda(), x_pcre_100.cuda()

    pad_mask_p_2000, pad_mask_p_500, pad_mask_p_100 = pad_mask_p_2000.cuda(), pad_mask_p_500.cuda(), pad_mask_p_100.cuda() 
    pad_mask_pcre_2000, pad_mask_pcre_500, pad_mask_pcre_100 = pad_mask_pcre_2000.cuda(), pad_mask_pcre_500.cuda(), pad_mask_pcre_100.cuda() 

    interaction_mask_2000, interaction_mask_500, interaction_mask_100 = interaction_mask_2000.cuda(), interaction_mask_500.cuda(), interaction_mask_100.cuda()
    interaction_freq = interaction_freq.cuda()

    out = model(
        x_p_2000, pad_mask_p_2000, x_pcre_2000, pad_mask_pcre_2000, interaction_mask_2000,
        x_p_500, pad_mask_p_500, x_pcre_500, pad_mask_pcre_500, interaction_mask_500,
        x_p_100, pad_mask_p_100, x_pcre_100, pad_mask_pcre_100, interaction_mask_100,
        interaction_freq,
    )
