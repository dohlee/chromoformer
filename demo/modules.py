import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb, n_heads, d_model, idx, bias=True, scale=True, gate=False):
        super(MultiHeadAttention, self).__init__()

        d_head = d_model // n_heads
        self.n_heads, self.d_head = n_heads, d_head
        self.bias, self.scale = bias, scale
        
        self.gamma_f = nn.Parameter(torch.ones([n_heads]))

        self.w_bias = nn.Linear(2, n_heads, bias=False)
        self.gate = gate
        if gate:
            self.att = nn.Linear(d_emb, 4 * n_heads * d_head, bias=False)  # LinearNoBias for attention
        else:
            self.att = nn.Linear(d_emb, 3 * n_heads * d_head, bias=False)  # LinearNoBias for attention
        self.ff = nn.Linear(n_heads * d_head, d_emb, bias=bias)
        self.ln = nn.LayerNorm(d_emb)
        self.idx = idx

    def forward(self, x, mask=None, bias=None):
        ff_out = self.ff(self._attention(x, mask=mask, bias=bias))
        return self.ln(x + ff_out)

    def _attention(self, x, mask, bias):
        # x : bsz x n_ints x d_model
        bsz, n_ints = x.size(0), x.size(1)

        # self.att(x) : bsz x n_ints x (4 * n_heads * d_head)
        if self.gate:
            wq, wk, wv, gat = torch.chunk(self.att(x), 4, dim=-1) 
            # --> wq, wk, wv : bsz x n_ints x (n_heads * d_head)
            wq, wk, wv, gat = map(lambda x: x.view(bsz, n_ints, self.n_heads, self.d_head), (wq, wk, wv, gat))
            # --> wq, wk, wv : bsz x n_ints x n_heads x d_head
            wq, wk, wv, gat = wq.permute(0, 2, 1, 3), wk.permute(0, 2, 3, 1), wv.permute(0, 2, 1, 3), gat.permute(0, 2, 1, 3)
            # --> wq : bsz x n_heads x n_ints x d_head
            # --> wk : bsz x n_heads x d_head x n_ints
            # --> wv : bsz x n_heads x n_ints x d_head
            # --> gat : bsz x n_heads x n_ints x d_head
        else:
            wq, wk, wv = torch.chunk(self.att(x), 3, dim=-1) 
            # --> wq, wk, wv : bsz x n_ints x (n_heads * d_head)
            wq, wk, wv = map(lambda x: x.view(bsz, n_ints, self.n_heads, self.d_head), (wq, wk, wv))
            # --> wq, wk, wv : bsz x n_ints x n_heads x d_head
            wq, wk, wv = wq.permute(0, 2, 1, 3), wk.permute(0, 2, 3, 1), wv.permute(0, 2, 1, 3)
            # --> wq : bsz x n_heads x n_ints x d_head
            # --> wk : bsz x n_heads x d_head x n_ints
            # --> wv : bsz x n_heads x n_ints x d_head
            # --> gat : bsz x n_heads x n_ints x d_head

        att_score = torch.matmul(wq, wk)
        # --> att_score : bsz x n_heads x n_ints x n_ints
        if self.scale:
            att_score.div_(self.d_head ** 0.5)
        
        if bias is not None:
            # --> bias : bsz x n_ints x n_ints
            bias = bias.unsqueeze(1).expand(bsz, self.n_heads, n_ints, n_ints)
            # --> bias : bsz x n_heads x n_ints x n_ints
            gamma_f = self.gamma_f.unsqueeze(1).unsqueeze(2).expand(self.n_heads, n_ints, n_ints).unsqueeze(0).expand(bsz, self.n_heads, n_ints, n_ints)
            # --> gamma_f = bsz x n_heads x n_ints x n_ints
            att_score += gamma_f * bias

        if mask is not None:
            minus_inf = -65504 if att_score.dtype == torch.float16 else -1e9
            att_score = att_score.masked_fill(mask, minus_inf).type_as(att_score)
        att_prob = F.softmax(att_score, dim=-1)
        # --> att_prob : bsz x n_heads x n_ints x n_ints

        att_vec = torch.matmul(att_prob, wv)
        # --> att_vec : bsz x n_heads x n_ints x d_head

        if self.gate:
            att_vec = att_vec * torch.sigmoid(gat)
            att_vec = att_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bsz, n_ints, -1)
            # --> att_vec : bsz x n_ints x (h_heads * d_head)
        else:
            att_vec = att_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bsz, n_ints, -1)
            # --> att_vec : bsz x n_ints x (h_heads * d_head)

        return att_vec


class FeedForward(nn.Module):
    def __init__(self, d_emb, d_ff, activation):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_emb, d_ff)
        self.l2 = nn.Linear(d_ff, d_emb)
        self.ln = nn.LayerNorm(d_emb)

        self.act = activation

    def forward(self, x):
        return self.ln(x + self.l2(self.act(self.l1(x))))


class AttentionBlock(nn.Module):
    def __init__(self, d_emb, n_heads, d_model, d_ff, idx, activation, gate, bias=True, scale=True):
        super(AttentionBlock, self).__init__()
        self.self_att = MultiHeadAttention(d_emb, n_heads, d_model, idx, bias, scale, gate=gate)
        self.ff = FeedForward(d_emb, d_ff, activation=activation)
    
    def forward(self, x, mask=None, bias=None):
        return self.ff(self.self_att(x, mask, bias))


class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_emb, d_model, d_ff, activation=F.relu, gate=False):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(d_emb, n_heads, d_model, d_ff, idx=i, activation=activation, gate=gate) for i in range(n_layers)
        ])

    def forward(self, x, mask, bias=None):
        for layer in self.layers:
            x = layer(x, mask=mask, bias=bias)
        return x


class PairwiseMultiHeadAttention(nn.Module):
    def __init__(self, d_emb, n_heads, d_model, idx, res_p=0.0, att_p=0.0, bias=True, scale=True, gate=False):
        super(PairwiseMultiHeadAttention, self).__init__()

        d_head = d_model // n_heads
        self.n_heads, self.d_head = n_heads, d_head
        self.bias, self.scale = bias, scale
        
        self.gamma_f = nn.Parameter(torch.ones([n_heads]))
        self.gate = gate

        # Promoter -> Q
        self.p_att = nn.Linear(d_emb, n_heads * d_head, bias=False)
        # pCRE -> K, V
        self.c_att = nn.Linear(d_emb, 2 * n_heads * d_head, bias=False)

        self.ff = nn.Linear(n_heads * d_head, d_emb, bias=bias)

        self.drop_att, self.drop_res = nn.Dropout(att_p), nn.Dropout(res_p)
        self.ln = nn.LayerNorm(d_emb)

        self.idx = idx

    def forward(self, x_p, x_pcre, mask=None, bias=None):
        ff_out = self.ff(self._attention(x_p, x_pcre, mask=mask, bias=bias))
        return self.ln(x_p + self.drop_res(ff_out))

    def _attention(self, x_p, x_pcre, mask, bias):
        # x : bsz x n_ints x d_model
        bsz, n_ints = x_p.size(0), x_p.size(1)
        # self.att(x) : bsz x n_ints x (4 * n_heads * d_head)

        wq = self.p_att(x_p)
        wk, wv = torch.chunk(self.c_att(x_pcre), 2, dim=-1) 
        
        # --> wq, wk, wv : bsz x n_ints x (n_heads * d_head)
        wq, wk, wv = map(lambda x: x.view(bsz, n_ints, self.n_heads, self.d_head), (wq, wk, wv))
        # --> wq, wk, wv : bsz x n_ints x n_heads x d_head
        wq, wk, wv = wq.permute(0, 2, 1, 3), wk.permute(0, 2, 3, 1), wv.permute(0, 2, 1, 3)
        # --> wq : bsz x n_heads x n_ints x d_head
        # --> wk : bsz x n_heads x d_head x n_ints
        # --> wv : bsz x n_heads x n_ints x d_head

        att_score = torch.matmul(wq, wk)
        # --> att_score : bsz x n_heads x n_ints x n_ints
        if self.scale:
            att_score.div_(self.d_head ** 0.5)
        
        if bias is not None:
            # --> bias : bsz x n_ints x n_ints
            bias = bias.unsqueeze(1).expand(bsz, self.n_heads, n_ints, n_ints)
            # --> bias : bsz x n_heads x n_ints x n_ints
            gamma_f = self.gamma_f.unsqueeze(1).unsqueeze(2).expand(self.n_heads, n_ints, n_ints).unsqueeze(0).expand(bsz, self.n_heads, n_ints, n_ints)
            # --> gamma_f = bsz x n_heads x n_ints x n_ints
            att_score += gamma_f * bias

        if mask is not None:
            minus_inf = -65504 if att_score.dtype == torch.float16 else -1e9
            att_score = att_score.masked_fill(mask, minus_inf).type_as(att_score)
        att_prob = self.drop_att(F.softmax(att_score, dim=-1))
        # --> att_prob : bsz x n_heads x n_ints x n_ints

        att_vec = torch.matmul(att_prob, wv)
        # --> att_vec : bsz x n_heads x n_ints x d_head

        att_vec = att_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bsz, n_ints, -1)
        # --> att_vec : bsz x n_ints x (h_heads * d_head)

        return att_vec


class PairwiseAttentionBlock(nn.Module):
    def __init__(self, d_emb, n_heads, d_model, d_ff, idx, activation, gate, res_p=0.0, att_p=0.0, bias=True, scale=True):
        super(PairwiseAttentionBlock, self).__init__()

        self.self_att = PairwiseMultiHeadAttention(d_emb, n_heads, d_model, idx, res_p, att_p, bias, scale, gate=gate)
        self.ff = FeedForward(d_emb, d_ff, activation=activation)
    
    def forward(self, x_p, x_pcre, mask=None, bias=None):
        return self.ff(self.self_att(x_p, x_pcre, mask, bias))


class PairwiseTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_emb, d_model, d_ff, res_p=0.0, att_p=0.0, activation=F.relu, gate=False):
        super(PairwiseTransformer, self).__init__()

        self.layers = nn.ModuleList([
            PairwiseAttentionBlock(d_emb, n_heads, d_model, d_ff, idx=i, res_p=res_p, att_p=att_p, activation=activation, gate=gate) for i in range(n_layers)
        ])

    def forward(self, x_p, x_pcre, mask, bias=None):
        for layer in self.layers:
            x_p = layer(x_p, x_pcre, mask=mask, bias=bias)
        return x_p
