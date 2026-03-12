import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class SparseVariableRouter(nn.Module):
    def __init__(self, num_vars, hidden_dim=16, topk=8, temp=1.0):
        super(SparseVariableRouter, self).__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.temp = temp

        self.var_embed = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(self.var_embed)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def forward(self, x):
        B, L, N = x.shape
        device = x.device

        var_emb = self.var_embed.expand(B, -1, -1)
        Q = self.query_proj(var_emb)
        K = self.key_proj(var_emb)
        sim_matrix = torch.bmm(Q, K.transpose(1, 2))

        mask = torch.eye(N, device=device).bool().unsqueeze(0)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(sim_matrix) + 1e-10) + 1e-10)
            gumbel_sim = sim_matrix + gumbel_noise
        else:
            gumbel_sim = sim_matrix

        topk_vals, topk_indices = torch.topk(gumbel_sim, self.topk, dim=-1)
        sparse_weights = F.softmax(topk_vals / self.temp, dim=-1)

        x_perm = x.permute(0, 2, 1)
        x_expanded = x_perm.unsqueeze(1).expand(-1, N, -1, -1)
        neighbors = torch.gather(
            x_expanded, dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, x_perm.shape[-1])
        )

        weighted = neighbors * sparse_weights.unsqueeze(-1)
        aggregated = weighted.sum(dim=2)
        return aggregated.permute(0, 2, 1)


class UnifiedRingStarBlock(nn.Module):
    def __init__(self, d_core=64, kernel_size=3, dilation=1):
        super(UnifiedRingStarBlock, self).__init__()
        self.d_core = d_core
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.variable_router = None

        self.ring_conv = None
        self.center_proj = None
        self.center_to_n = None
        self.gate_layer = None
        self.fusion_proj = None
        self.norm = None
        self.score_proj = None

    def forward(self, x):
        B, L, N = x.shape
        device = x.device

        if self.ring_conv is None:
            padding = (self.kernel_size - 1) // 2
            self.ring_conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(self.kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular',
                bias=False,
            ).to(device)

        if self.variable_router is None:
            self.variable_router = SparseVariableRouter(num_vars=N, topk=8, hidden_dim=16).to(device)
        ring_out = self.variable_router(x)

        if self.center_proj is None:
            self.center_proj = nn.Sequential(
                nn.Linear(N, self.d_core).to(device),
                nn.GELU(),
                nn.Linear(self.d_core, self.d_core).to(device),
                nn.GELU()
            )
            self.score_proj = nn.Linear(N, 1).to(device)
            self.center_to_n = nn.Linear(self.d_core, N).to(device)

        attn_scores = self.score_proj(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        center_raw = torch.sum(x * attn_weights, dim=1)
        center_vec = self.center_proj(center_raw)
        center_vec = self.center_to_n(center_vec)
        center_broadcast = center_vec.unsqueeze(1).expand(-1, L, -1)

        if self.gate_layer is None:
            self.gate_layer = nn.Linear(N * 2, N).to(device)
            self.fusion_proj = nn.Linear(N, N).to(device)
        if self.norm is None:
            self.norm = nn.LayerNorm(N).to(device)

        fusion_input = torch.cat([ring_out, center_broadcast], dim=-1)
        gate = torch.sigmoid(self.gate_layer(fusion_input))
        fused = gate * ring_out + (1 - gate) * center_broadcast
        fused = self.fusion_proj(fused)
        out = self.norm(fused + x)

        return out, None


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        self.encoder = Encoder([
            EncoderLayer(
                UnifiedRingStarBlock(
                    d_core=configs.d_core,
                    kernel_size=getattr(configs, 'ring_kernel', 3)
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ])

        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = self.layer_norm(enc_out)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
