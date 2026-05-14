from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RWKV7CharLMConfig:
    vocab_size: int
    n_layer: int = 16
    n_embd: int = 64
    head_size: int = 16

    @property
    def dim_att(self) -> int:
        return self.n_embd

    @property
    def dim_ffn(self) -> int:
        return 4 * self.n_embd

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.head_size <= 0:
            raise ValueError("head_size must be positive")
        if self.n_embd % self.head_size != 0:
            raise ValueError("n_embd must be divisible by head_size")


def _time_shift_delta(x: Tensor) -> Tensor:
    return F.pad(x[:, :-1], (0, 0, 1, 0)) - x


def _ortho_init(weight: Tensor, scale: float) -> Tensor:
    with torch.no_grad():
        shape = weight.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(weight, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for idx in range(shape[0]):
                nn.init.orthogonal_(weight[idx], gain=gain * scale)
        else:
            raise ValueError("orthogonal init expects rank 2 or 3")
    return weight


def rwkv7_scan(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, *, head_size: int) -> Tensor:
    """Reference RWKV-7 recurrence from the v7 demo, intentionally plain PyTorch."""
    batch, length, channels = r.shape
    heads = channels // head_size
    r = r.view(batch, length, heads, head_size).float()
    k = k.view(batch, length, heads, head_size).float()
    v = v.view(batch, length, heads, head_size).float()
    a = a.view(batch, length, heads, head_size).float()
    b = b.view(batch, length, heads, head_size).float()
    w = torch.exp(-torch.exp(w.view(batch, length, heads, head_size).float()))
    state = torch.zeros(batch, heads, head_size, head_size, device=r.device, dtype=torch.float)
    outputs = []
    for step in range(length):
        kk = k[:, step].view(batch, heads, 1, head_size)
        rr = r[:, step].view(batch, heads, head_size, 1)
        vv = v[:, step].view(batch, heads, head_size, 1)
        aa = a[:, step].view(batch, heads, head_size, 1)
        bb = b[:, step].view(batch, heads, 1, head_size)
        state = state * w[:, step, :, None, :] + state @ aa @ bb + vv @ kk
        outputs.append((state @ rr).view(batch, heads, head_size))
    return torch.stack(outputs, dim=1).view(batch, length, channels).to(dtype=r.dtype)


class RWKV7TimeMix(nn.Module):
    def __init__(self, config: RWKV7CharLMConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.head_size = config.head_size
        self.n_head = config.n_embd // config.head_size
        heads = self.n_head
        head_size = self.head_size
        channels = config.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(1, config.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - layer_id / config.n_layer
            ddd = torch.ones(1, 1, channels)
            for idx in range(channels):
                ddd[0, 0, idx] = idx / channels
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            www = torch.zeros(channels)
            zigzag = torch.zeros(channels)
            linear = torch.zeros(channels)
            for idx in range(channels):
                linear[idx] = idx / max(1, channels - 1) - 0.5
                zigzag[idx] = ((idx % head_size) - ((head_size - 1) / 2)) / ((head_size - 1) / 2)
                zigzag[idx] = zigzag[idx] * abs(zigzag[idx])
                www[idx] = -6 + 6 * (idx / max(1, channels - 1)) ** (1 + ratio_0_to_1**0.3)

            decay_lora = 8
            aaa_lora = 8
            mv_lora = 8
            gate_lora = 8
            self.w1 = nn.Parameter(torch.zeros(channels, decay_lora))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(decay_lora, channels), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, channels) + 0.5 + zigzag * 2.5)
            self.a1 = nn.Parameter(torch.zeros(channels, aaa_lora))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(aaa_lora, channels), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, channels) - 0.19 + zigzag * 0.3 + linear * 0.4)
            self.v1 = nn.Parameter(torch.zeros(channels, mv_lora))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(mv_lora, channels), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, channels) + 0.73 - linear * 0.4)
            self.g1 = nn.Parameter(torch.zeros(channels, gate_lora))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(gate_lora, channels), 0.1))
            self.k_k = nn.Parameter(torch.zeros(1, 1, channels) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, channels) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(heads, head_size) - 0.04)

        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.ln_x = nn.GroupNorm(heads, channels, eps=64e-5)
        self.reset_parameters(channels)

    def reset_parameters(self, channels: int) -> None:
        self.receptance.weight.data.uniform_(-0.5 / (channels**0.5), 0.5 / (channels**0.5))
        self.key.weight.data.uniform_(-0.05 / (channels**0.5), 0.05 / (channels**0.5))
        self.value.weight.data.uniform_(-0.5 / (channels**0.5), 0.5 / (channels**0.5))
        self.output.weight.data.zero_()

    def forward(self, x: Tensor, v_first: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, channels = x.shape
        heads = self.n_head
        xx = _time_shift_delta(x)
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(batch, length, heads, -1), dim=-1, p=2.0).view(batch, length, channels)
        k = k * (1 + (a - 1) * self.k_a)
        out = rwkv7_scan(r, w, k, v, -kk, kk * a, head_size=self.head_size)
        out = self.ln_x(out.view(batch * length, channels)).view(batch, length, channels)
        out = out + ((r.view(batch, length, heads, -1) * k.view(batch, length, heads, -1) * self.r_k).sum(
            dim=-1, keepdim=True
        ) * v.view(batch, length, heads, -1)).view(batch, length, channels)
        return self.output(out * g), v_first


class RWKV7ChannelMix(nn.Module):
    def __init__(self, config: RWKV7CharLMConfig) -> None:
        super().__init__()
        channels = config.n_embd
        self.x_k = nn.Parameter(torch.zeros(1, 1, channels))
        self.key = nn.Linear(channels, config.dim_ffn, bias=False)
        self.value = nn.Linear(config.dim_ffn, channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.value.weight.data.zero_()
            nn.init.orthogonal_(self.key.weight.data, gain=2.0)

    def forward(self, x: Tensor) -> Tensor:
        xx = _time_shift_delta(x)
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class RWKV7Block(nn.Module):
    def __init__(self, config: RWKV7CharLMConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(config.n_embd)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.att = RWKV7TimeMix(config, layer_id)
        self.ffn = RWKV7ChannelMix(config)

    def forward(self, x: Tensor, v_first: Tensor) -> tuple[Tensor, Tensor]:
        if self.layer_id == 0:
            x = self.ln0(x)
        att_out, v_first = self.att(self.ln1(x), v_first)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7CharLM(nn.Module):
    """RWKV-7 x070-style char LM based on the official RWKV-v7 reference."""

    def __init__(self, config: RWKV7CharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RWKV7Block(config, layer_id) for layer_id in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        x = self.emb(input_ids)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.head(self.ln_out(x))
