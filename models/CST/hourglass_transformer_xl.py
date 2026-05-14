import sys
import math
import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import CausalSSMBranch

_ROOT = Path(__file__).resolve().parents[2]
_TXL_UTILS = _ROOT / "external_references" / "transformer-xl" / "pytorch" / "utils"
if str(_TXL_UTILS) not in sys.path:
    sys.path.append(str(_TXL_UTILS))
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits


@dataclass
class HourglassTransformerXL2xOutput:
    loss: torch.Tensor
    pair_logits: torch.Tensor
    pair_targets: torch.Tensor
    pair_target_mask: torch.Tensor
    compressed_times: torch.Tensor
    projected_down: torch.Tensor


@dataclass
class XLHourglass2XDownToUpConnectingBlockOutput:
    main_h: torch.Tensor
    main_mem: torch.Tensor
    cross_h: torch.Tensor
    cross_mem: torch.Tensor
    kv_states: torch.Tensor
    kv_times: torch.Tensor
    cross_mask: torch.Tensor


def _normalize_causal_times(times, *, expected_len, device):
    if times is None:
        return None
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.long, device=device)
    else:
        times = times.to(device=device, dtype=torch.long)
    if times.ndim != 1:
        raise ValueError(f"causal times must be rank 1, got shape {tuple(times.shape)}")
    if times.numel() != expected_len:
        raise ValueError(f"causal times length {times.numel()} does not match expected length {expected_len}")
    return times


def build_hourglass_causal_attn_mask(
    *,
    qlen,
    klen,
    mlen,
    device,
    same_length=False,
    mem_len=0,
    query_times=None,
    key_times=None,
):
    """Return a TXL-style mask where True means the key is hidden from the query."""
    query_times = _normalize_causal_times(query_times, expected_len=qlen, device=device)
    key_times = _normalize_causal_times(key_times, expected_len=klen, device=device)
    if query_times is not None or key_times is not None:
        if query_times is None or key_times is None:
            raise ValueError("query_times and key_times must be provided together")
        return (key_times[None, :] > query_times[:, None])[:, :, None]

    all_ones = torch.ones(qlen, klen, device=device, dtype=torch.bool)
    if same_length:
        mask_len = klen - mem_len
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        return (torch.triu(all_ones, 1 + mlen) | torch.tril(all_ones, -mask_shift_len))[:, :, None]
    return torch.triu(all_ones, diagonal=1 + mlen)[:, :, None]

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class HymbaRelPartialLearnableDecoderLayer(nn.Module):
    """Transformer-XL relative attention layer with a causal SSM side branch."""

    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        *,
        ssm_kernel_size=3,
        **kwargs,
    ):
        super(HymbaRelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            **kwargs,
        )
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = CausalSSMBranch(d_model, conv_kernel_size=ssm_kernel_size)
        self.ssm_drop = nn.Dropout(dropout)
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=kwargs.get("pre_lnorm"),
        )

    def forward(
        self,
        dec_inp,
        r,
        r_w_bias,
        r_r_bias,
        dec_attn_mask=None,
        mems=None,
        causal_times=None,
    ):
        attn_output = self.dec_attn(
            dec_inp,
            r,
            r_w_bias,
            r_r_bias,
            attn_mask=dec_attn_mask,
            mems=mems,
        )
        if causal_times is None:
            causal_times = torch.arange(dec_inp.size(0), device=dec_inp.device, dtype=torch.long)
        causal_times = _normalize_causal_times(
            causal_times,
            expected_len=dec_inp.size(0),
            device=dec_inp.device,
        )
        ssm_input = self.ssm_norm(dec_inp).transpose(0, 1).contiguous()
        ssm_output = self.ssm(ssm_input, causal_times=causal_times).transpose(0, 1).contiguous()
        fused = attn_output + 0.5 * self.ssm_drop(ssm_output)
        return self.pos_ff(fused)


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, 
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len
        self.last_dec_attn_mask = None
        self.last_query_times = None
        self.last_key_times = None

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, query_times=None, key_times=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        dec_attn_mask = build_hourglass_causal_attn_mask(
            qlen=qlen,
            klen=klen,
            mlen=mlen,
            device=word_emb.device,
            same_length=self.same_length,
            mem_len=self.mem_len,
            query_times=query_times,
            key_times=key_times,
        )
        self.last_dec_attn_mask = dec_attn_mask.detach()
        self.last_query_times = None if query_times is None else _normalize_causal_times(
            query_times, expected_len=qlen, device=word_emb.device
        ).detach()
        self.last_key_times = None if key_times is None else _normalize_causal_times(
            key_times, expected_len=klen, device=word_emb.device
        ).detach()

        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems, query_times=None, key_times=None):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems, query_times=query_times, key_times=key_times)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


class HourglassTransformerXL(MemTransformerLM):
    """Editable Transformer-XL copy with causal-time masks for compressed streams."""

    pass


class HymbaTransformerXL(MemTransformerLM):
    """Transformer-XL baseline that swaps each decoder layer for TXL attention + SSM."""

    def __init__(self, *args, ssm_kernel_size=3, **kwargs):
        super(HymbaTransformerXL, self).__init__(*args, **kwargs)
        if self.attn_type != 0:
            raise ValueError("HymbaTransformerXL currently supports attn_type=0 only")
        self.ssm_kernel_size = ssm_kernel_size
        self.layers = nn.ModuleList(
            [
                HymbaRelPartialLearnableDecoderLayer(
                    self.n_head,
                    self.d_model,
                    self.d_head,
                    kwargs["d_inner"] if "d_inner" in kwargs else args[5],
                    kwargs["dropout"] if "dropout" in kwargs else args[6],
                    tgt_len=self.tgt_len,
                    ext_len=self.ext_len,
                    mem_len=self.mem_len,
                    dropatt=kwargs["dropatt"] if "dropatt" in kwargs else args[7],
                    pre_lnorm=kwargs.get("pre_lnorm", False),
                    ssm_kernel_size=ssm_kernel_size,
                )
                for _ in range(self.n_layer)
            ]
        )


class XLHourglass2XDownToUpConnectingBlock(nn.Module):
    """Half-rate TXL block that prepares skipped phase information for upsample."""

    def __init__(
        self,
        *,
        d_model,
        n_head,
        d_head,
        d_inner,
        dropout=0.0,
        dropatt=0.0,
        pre_lnorm=True,
        clamp_len=-1,
    ):
        super(XLHourglass2XDownToUpConnectingBlock, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.clamp_len = clamp_len
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.layer = RelPartialLearnableDecoderLayer(
            n_head,
            d_model,
            d_head,
            d_inner,
            dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
        )
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)
        self.last_cross_query_times = None
        self.last_cross_key_times = None
        self.last_cross_mask = None
        self.last_kv_times = None

    def _validate_stream(self, name, states, times):
        if states.ndim != 3:
            raise ValueError(f"{name} states must be [seq, batch, dim], got {tuple(states.shape)}")
        if states.size(-1) != self.d_model:
            raise ValueError(f"{name} dim {states.size(-1)} != d_model {self.d_model}")
        times = _normalize_causal_times(times, expected_len=states.size(0), device=states.device)
        return times

    def _pos_emb(self, klen, ref):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=ref.device, dtype=ref.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        return self.drop(self.pos_emb(pos_seq))

    def forward(
        self,
        *,
        main_h,
        main_h_times,
        main_mem,
        main_mem_times,
        cross_h,
        cross_h_times,
        cross_mem,
        cross_mem_times,
    ):
        main_h_times = self._validate_stream("main_h", main_h, main_h_times)
        main_mem_times = self._validate_stream("main_mem", main_mem, main_mem_times)
        cross_h_times = self._validate_stream("cross_h", cross_h, cross_h_times)
        cross_mem_times = self._validate_stream("cross_mem", cross_mem, cross_mem_times)
        for name, states in (
            ("main_mem", main_mem),
            ("cross_h", cross_h),
            ("cross_mem", cross_mem),
        ):
            if states.size(1) != main_h.size(1):
                raise ValueError(f"{name} batch size must match main_h")

        cross_key_times = torch.cat([cross_mem_times, cross_h_times], dim=0)
        cross_mask = build_hourglass_causal_attn_mask(
            qlen=cross_h.size(0),
            klen=cross_mem.size(0) + cross_h.size(0),
            mlen=cross_mem.size(0),
            device=cross_h.device,
            query_times=cross_h_times,
            key_times=cross_key_times,
        )
        pos_emb = self._pos_emb(cross_mem.size(0) + cross_h.size(0), cross_h)
        updated_cross_h = self.layer(
            cross_h,
            pos_emb,
            self.r_w_bias,
            self.r_r_bias,
            dec_attn_mask=cross_mask,
            mems=cross_mem if cross_mem.numel() else None,
        )
        kv_states = torch.cat([main_h, main_mem, updated_cross_h, cross_mem], dim=0)
        kv_times = torch.cat([main_h_times, main_mem_times, cross_h_times, cross_mem_times], dim=0)

        self.last_cross_query_times = cross_h_times.detach()
        self.last_cross_key_times = cross_key_times.detach()
        self.last_cross_mask = cross_mask.detach()
        self.last_kv_times = kv_times.detach()
        return XLHourglass2XDownToUpConnectingBlockOutput(
            main_h=main_h,
            main_mem=main_mem,
            cross_h=updated_cross_h,
            cross_mem=cross_mem,
            kv_states=kv_states,
            kv_times=kv_times,
            cross_mask=cross_mask,
        )


class HourglassTransformerXL2x(MemTransformerLM):
    """Transformer-XL hourglass with one 1x -> 2x transition and pair-token loss."""

    def __init__(
        self,
        *args,
        transition_layer=8,
        pair_dim=None,
        pair_target_offset=0,
        transition_mode="alternate_h_mem",
        **kwargs,
    ):
        init_d_inner = kwargs.get("d_inner", args[5] if len(args) > 5 else None)
        init_dropout = kwargs.get("dropout", args[6] if len(args) > 6 else 0.0)
        init_dropatt = kwargs.get("dropatt", args[7] if len(args) > 7 else 0.0)
        init_pre_lnorm = kwargs.get("pre_lnorm", False)
        init_clamp_len = kwargs.get("clamp_len", -1)
        super(HourglassTransformerXL2x, self).__init__(*args, **kwargs)
        if self.attn_type != 0:
            raise ValueError("HourglassTransformerXL2x currently supports attn_type=0 only")
        if transition_layer <= 0 or transition_layer >= self.n_layer:
            raise ValueError("transition_layer must split the stack into full-rate and compressed layers")
        self.transition_layer = transition_layer
        self.pair_dim = pair_dim or (2 * self.d_model)
        if pair_target_offset < 0:
            raise ValueError("pair_target_offset must be nonnegative")
        self.pair_target_offset = pair_target_offset
        if transition_mode not in (
            "alternate_h_mem",
            "sum_h_pair_with_on_mem",
            "connecting_block",
            "connecting_block_sum_h_pair",
        ):
            raise ValueError(f"unknown transition_mode: {transition_mode}")
        self.transition_mode = transition_mode
        if transition_mode in ("connecting_block", "connecting_block_sum_h_pair"):
            self.down_to_up_connector = XLHourglass2XDownToUpConnectingBlock(
                d_model=self.d_model,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=init_d_inner,
                dropout=init_dropout,
                dropatt=init_dropatt,
                pre_lnorm=init_pre_lnorm,
                clamp_len=init_clamp_len,
            )
            self.connector_injection_layer = RelPartialLearnableDecoderLayer(
                self.n_head,
                self.d_model,
                self.d_head,
                init_d_inner,
                init_dropout,
                dropatt=init_dropatt,
                pre_lnorm=init_pre_lnorm,
            )
        self.pair_up = nn.Linear(self.d_model, self.pair_dim)
        self.pair_head = nn.Linear(self.pair_dim, 2 * self.n_token)
        self.pair_down = nn.Linear(self.pair_dim, self.d_model)
        self.last_transition_query_times = None
        self.last_transition_key_times = None
        self.last_transition_mask = None
        self.last_compressed_masks = []
        self.last_compressed_query_times = []
        self.last_compressed_key_times = []
        self.last_connector_kv_times = None
        self.last_connector_injection_mask = None

    def _attn0_pos_emb(self, klen, word_emb):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        return self.drop(self.pos_emb(pos_seq))

    def _attn0_layer(self, layer_idx, core_out, pos_emb, dec_attn_mask, mems_i=None):
        return self.layers[layer_idx](
            core_out,
            pos_emb,
            self.r_w_bias,
            self.r_r_bias,
            dec_attn_mask=dec_attn_mask,
            mems=mems_i,
        )

    def _pair_targets(self, target):
        pair_slots = (target.size(0) + 1) // 2
        pair_targets = target.new_zeros(pair_slots, target.size(1), 2)
        pair_mask = torch.zeros(pair_slots, target.size(1), 2, dtype=torch.bool, device=target.device)

        first_targets = target[self.pair_target_offset::2]
        pair_targets[: first_targets.size(0), :, 0] = first_targets
        pair_mask[: first_targets.size(0), :, 0] = True

        second_targets = target[self.pair_target_offset + 1::2]
        if second_targets.numel():
            pair_targets[: second_targets.size(0), :, 1] = second_targets
            pair_mask[: second_targets.size(0), :, 1] = True
        return pair_targets, pair_mask

    def _pair_loss(self, pair_logits, pair_targets, pair_mask):
        flat_logits = pair_logits[pair_mask]
        flat_targets = pair_targets[pair_mask]
        if flat_targets.numel() == 0:
            raise ValueError("pair target mask must select at least one target")
        return F.cross_entropy(flat_logits, flat_targets)

    def _forward_2x(self, dec_inp):
        qlen, _bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        source_times = torch.arange(qlen, device=word_emb.device, dtype=torch.long)

        core_out = self.drop(word_emb)
        for layer_idx in range(self.transition_layer):
            mask = build_hourglass_causal_attn_mask(
                qlen=core_out.size(0),
                klen=core_out.size(0),
                mlen=0,
                device=core_out.device,
                query_times=source_times,
                key_times=source_times,
            )
            pos_emb = self._attn0_pos_emb(core_out.size(0), word_emb)
            core_out = self._attn0_layer(layer_idx, core_out, pos_emb, mask)

        off_h_states = core_out[0::2]
        off_h_times = source_times[0::2]
        on_h_states = core_out[1::2]
        on_h_times = source_times[1::2]
        connector_output = None
        if self.transition_mode == "alternate_h_mem":
            h_states = off_h_states
            h_times = off_h_times
            mem_states = on_h_states
            mem_times = on_h_times
        elif self.transition_mode == "sum_h_pair_with_on_mem":
            h_states = off_h_states.clone()
            h_times = off_h_times
            mem_states = on_h_states
            mem_times = on_h_times
            if on_h_states.numel():
                h_states[: on_h_states.size(0)] = h_states[: on_h_states.size(0)] + on_h_states
        elif self.transition_mode == "connecting_block":
            h_states = off_h_states
            h_times = off_h_times
            mem_states = on_h_states
            mem_times = on_h_times
            connector_output = self.down_to_up_connector(
                main_h=off_h_states,
                main_h_times=off_h_times,
                main_mem=on_h_states,
                main_mem_times=on_h_times,
                cross_h=on_h_states,
                cross_h_times=on_h_times,
                cross_mem=off_h_states,
                cross_mem_times=off_h_times,
            )
        else:
            h_states = off_h_states.clone()
            h_times = off_h_times
            mem_states = on_h_states
            mem_times = on_h_times
            paired_len = on_h_states.size(0)
            if paired_len:
                h_states[:paired_len] = off_h_states[:paired_len] + on_h_states
                cross_h_states = off_h_states[:paired_len] + on_h_states
            else:
                cross_h_states = on_h_states
            connector_output = self.down_to_up_connector(
                main_h=h_states,
                main_h_times=h_times,
                main_mem=on_h_states,
                main_mem_times=on_h_times,
                cross_h=cross_h_states,
                cross_h_times=on_h_times,
                cross_mem=off_h_states,
                cross_mem_times=off_h_times,
            )

        compressed_times = h_times.clone()
        if mem_times.numel():
            compressed_times[: mem_times.numel()] = torch.maximum(
                h_times[: mem_times.numel()],
                mem_times,
            )

        transition_key_times = torch.cat([mem_times, h_times], dim=0)
        transition_mask = build_hourglass_causal_attn_mask(
            qlen=h_states.size(0),
            klen=mem_states.size(0) + h_states.size(0),
            mlen=mem_states.size(0),
            device=core_out.device,
            query_times=compressed_times,
            key_times=transition_key_times,
        )
        transition_pos_emb = self._attn0_pos_emb(mem_states.size(0) + h_states.size(0), word_emb)
        core_out = self._attn0_layer(
            self.transition_layer,
            h_states,
            transition_pos_emb,
            transition_mask,
            mems_i=mem_states if mem_states.numel() else None,
        )
        self.last_transition_query_times = compressed_times.detach()
        self.last_transition_key_times = transition_key_times.detach()
        self.last_transition_mask = transition_mask.detach()
        if connector_output is not None:
            injection_key_times = torch.cat([connector_output.kv_times, compressed_times], dim=0)
            injection_mask = build_hourglass_causal_attn_mask(
                qlen=core_out.size(0),
                klen=connector_output.kv_states.size(0) + core_out.size(0),
                mlen=connector_output.kv_states.size(0),
                device=core_out.device,
                query_times=compressed_times,
                key_times=injection_key_times,
            )
            injection_pos_emb = self._attn0_pos_emb(connector_output.kv_states.size(0) + core_out.size(0), word_emb)
            core_out = self.connector_injection_layer(
                core_out,
                injection_pos_emb,
                self.r_w_bias,
                self.r_r_bias,
                dec_attn_mask=injection_mask,
                mems=connector_output.kv_states,
            )
            self.last_connector_kv_times = connector_output.kv_times.detach()
            self.last_connector_injection_mask = injection_mask.detach()
        else:
            self.last_connector_kv_times = None
            self.last_connector_injection_mask = None

        self.last_compressed_masks = []
        self.last_compressed_query_times = []
        self.last_compressed_key_times = []
        for layer_idx in range(self.transition_layer + 1, self.n_layer):
            compressed_mask = build_hourglass_causal_attn_mask(
                qlen=core_out.size(0),
                klen=core_out.size(0),
                mlen=0,
                device=core_out.device,
                query_times=compressed_times,
                key_times=compressed_times,
            )
            pos_emb = self._attn0_pos_emb(core_out.size(0), word_emb)
            core_out = self._attn0_layer(layer_idx, core_out, pos_emb, compressed_mask)
            self.last_compressed_masks.append(compressed_mask.detach())
            self.last_compressed_query_times.append(compressed_times.detach())
            self.last_compressed_key_times.append(compressed_times.detach())

        return self.drop(core_out), compressed_times

    def forward(self, data, target, *mems):
        if mems:
            raise ValueError("HourglassTransformerXL2x does not support recurrent mems yet")
        if data.shape != target.shape:
            raise ValueError("data and target must have the same [seq, batch] shape")

        hidden, compressed_times = self._forward_2x(data)
        pair_features = F.gelu(self.pair_up(hidden))
        pair_logits = self.pair_head(pair_features).view(
            hidden.size(0),
            hidden.size(1),
            2,
            self.n_token,
        )
        pair_targets, pair_mask = self._pair_targets(target)
        if pair_targets.shape != pair_logits.shape[:3]:
            raise AssertionError("pair targets must match pair logits [compressed_seq, batch, 2]")
        loss = self._pair_loss(pair_logits, pair_targets, pair_mask)
        projected_down = self.pair_down(pair_features)
        return HourglassTransformerXL2xOutput(
            loss=loss,
            pair_logits=pair_logits,
            pair_targets=pair_targets,
            pair_target_mask=pair_mask,
            compressed_times=compressed_times,
            projected_down=projected_down,
        )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True, 
                            d_embed=d_embed, div_val=div_val, 
                            tie_projs=tie_projs, pre_lnorm=True,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, 
                            cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
