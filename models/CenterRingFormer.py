import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class CenterRingFormerPlus(nn.Module):
    def __init__(self, d_series, num_jumps=2, long_jump_span=2, d_center=None, num_centers=4, seq_len=96):
        super(CenterRingFormerPlus, self).__init__()

        self.d_series = d_series
        self.num_jumps = num_jumps
        self.long_jump_span = long_jump_span
        self.num_centers = num_centers
        self.d_center = d_center if d_center is not None else d_series
        self.seq_len = seq_len

        # --- 可学习位置编码（增强拓扑感知） ---
        # self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_series))

        # --- Ring + Jump 融合模块 ---
        self.fuse_ring = nn.Sequential(
            nn.Linear((2 + 2 * num_jumps + 1) * d_series, d_series),
            nn.GELU(),
            nn.Linear(d_series, d_series)
        )

        # --- 多中心编码模块 ---
        self.to_centers = nn.Sequential(
            nn.Linear(d_series, self.d_center),
            nn.GELU(),
            nn.Linear(self.d_center, self.d_center)
        )
        self.centers = nn.Parameter(torch.randn(1, num_centers, self.d_center))

        # --- 融合中心输出（Gate控制） ---
        self.fuse_center = nn.Sequential(
            nn.Linear(d_series + self.d_center, d_series),
            nn.GELU(),
            nn.Linear(d_series, d_series)
        )

        self.gate = nn.Sequential(
            nn.Linear(d_series + self.d_center, d_series),
            nn.Sigmoid()
        )

    def forward(self, queries, keys=None, values=None, attn_mask=None, **kwargs):
        # 兼容 EncoderLayer 的调用方式
        x = queries  # 忽略 keys/values，因为我们是自注意力结构
        B, N, D = x.shape
        # x = x + self.pos_embed[:, :N, :]  # 加入位置嵌入

        # --- Step 1: Ring 结构交互 ---
        left = torch.roll(x, shifts=1, dims=1)
        right = torch.roll(x, shifts=-1, dims=1)
        jumps = []
        for i in range(1, self.num_jumps + 1):
            jumps.append(torch.roll(x, shifts=i * self.long_jump_span, dims=1))
            jumps.append(torch.roll(x, shifts=-i * self.long_jump_span, dims=1))
        fusion_input = torch.cat([left, right, x] + jumps, dim=-1)
        x_ring = self.fuse_ring(fusion_input)  # [B, N, D]

        # --- Step 2: 多中心表示提取 ---
        token_proj = self.to_centers(x_ring)  # [B, N, d_center]
        center_weights = torch.softmax(
            torch.einsum('bnd,bkd->bnk', token_proj, self.centers.repeat(B, 1, 1)), dim=-1
        )  # [B, N, K]
        weighted_centers = torch.einsum('bnk,bkd->bnd', center_weights,
                                        self.centers.repeat(B, 1, 1))  # [B, N, d_center]

        # --- Step 3: 中心反馈 + gate 控制 ---
        fusion_input = torch.cat([x_ring, weighted_centers], dim=-1)
        gate = self.gate(fusion_input)
        fusion_center = gate * self.fuse_center(fusion_input) + (1 - gate) * x_ring  # 残差形式融合

        return fusion_center, None  # 注意要返回 tuple


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.dropout
        )

        self.encoder = Encoder([
            EncoderLayer(
                CenterRingFormerPlus(
                    d_series=configs.d_model,
                    num_jumps=configs.num_jumps,
                    long_jump_span=configs.long_jump_span,
                    d_center=configs.d_center,
                    num_centers=configs.n_centers,
                    seq_len=configs.seq_len
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ])

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, D]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, L, D]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
