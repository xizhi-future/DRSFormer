import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


# 新的
class SparseVariableRouter(nn.Module):
    def __init__(self, num_vars, hidden_dim=16):
        super(SparseVariableRouter, self).__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        self.topk = max(16, int(math.log2(num_vars) * 3))
        self.static_topk = max(32, int(math.sqrt(num_vars) * 2))

        self.var_embed = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.temporal_proj = nn.Linear(2, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.importance_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

        self.static_mask = None
        nn.init.xavier_uniform_(self.var_embed)
        nn.init.xavier_uniform_(self.temporal_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def build_static_mask(self, x):
        with torch.no_grad():
            B, L, N = x.shape
            x_avg = x.mean(dim=0)
            x_std = x.std(dim=0) + 1e-5
            normed = x_avg / x_std
            sim = torch.matmul(normed.T, normed)
            sim.fill_diagonal_(-1e9)
            _, topk_idx = torch.topk(sim, self.static_topk, dim=-1)
            mask = torch.zeros(N, N, device=x.device)
            mask.scatter_(1, topk_idx, 1.0)
            self.static_mask = mask.bool()

    def forward(self, x):  # [B, L, N]
        B, L, N = x.shape
        if self.static_mask is None:
            self.build_static_mask(x)

        # [B, N, L]
        x_var = x.permute(0, 2, 1)
        static_emb = self.var_embed.expand(B, -1, -1)
        # 时间特征: 均值 & 标准差
        mean = x_var.mean(dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x_var, dim=-1, keepdim=True) + 1e-5)
        time_feat = torch.cat([mean, std], dim=-1)
        time_emb = self.temporal_proj(time_feat)
        # 融合静态 + 时间特征
        var_fused = torch.cat([static_emb, time_emb], dim=-1)
        importance = self.importance_gate(var_fused).squeeze(-1).unsqueeze(1)
        # Q, K
        Q = self.query_proj(var_fused)
        K = self.key_proj(var_fused)
        sim_matrix = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)  # [B, N, N]
        # 静态先验掩码
        mask_static = self.static_mask.unsqueeze(0).expand(B, -1, -1)
        sim_matrix = sim_matrix.masked_fill(~mask_static, -1e9)
        # --------------------------
        # 动态 topk 稀疏化 (在静态候选集合内再选 topk)
        # --------------------------
        topk_val, topk_idx = torch.topk(sim_matrix, self.topk, dim=-1)  # [B, N, topk]
        mask_dynamic = torch.zeros_like(sim_matrix, dtype=torch.bool)
        mask_dynamic.scatter_(2, topk_idx, True)
        sim_matrix = sim_matrix.masked_fill(~mask_dynamic, -1e9)
        # softmax 注意力
        attn_weights = F.softmax(sim_matrix, dim=-1)
        # 聚合
        routed = torch.bmm(attn_weights, x_var)  # [B, N, L]
        routed_out = routed.permute(0, 2, 1)  # [B, L, N]

        return importance * routed_out + (1 - importance) * x


# 新的
class HTMBlock(nn.Module):
    def __init__(self, num_vars, d_model, dropout=0.1):
        super(HTMBlock, self).__init__()
        self.num_vars = num_vars
        self.d_model = d_model

        # 分支1：趋势平滑
        self.t1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=7, stride=1, padding=3),
            nn.Conv1d(num_vars, num_vars, kernel_size=1),
            nn.BatchNorm1d(num_vars)
        )

        # ✅ 分支2：高效 depthwise 结构（group=32）
        self.t2 = nn.Sequential(
            nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm1d(num_vars),
            nn.GELU(),
            nn.Conv1d(num_vars, num_vars, kernel_size=1),
        )

        # 分支3：Res-TCN 分支（保留）
        self.t3 = nn.Sequential(
            nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars),
            nn.GELU(),
            nn.Conv1d(num_vars, num_vars, kernel_size=1),
            nn.BatchNorm1d(num_vars),
            nn.GELU(),
            nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars),
            nn.Conv1d(num_vars, num_vars, kernel_size=1)
        )

        # ✅ gate 输入简化（只用 mean/std + pred_len）
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * num_vars + 1, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        self.proj = nn.Conv1d(num_vars, num_vars, kernel_size=1)
        self.norm = nn.LayerNorm(num_vars)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, L, N]
        B, L, N = x.shape
        x_ = x.permute(0, 2, 1)  # [B, N, L]

        # 三个分支
        t1_out = self.t1(x_)
        t2_out = self.t2(x_)
        t3_out = self.t3(x_)

    # ============ 添加标准化操作 ============
        # 对每个分支输出进行LayerNorm标准化，统一尺度
        t1_out = F.layer_norm(t1_out, [t1_out.size(-1)])
        t2_out = F.layer_norm(t2_out, [t2_out.size(-1)])
        t3_out = F.layer_norm(t3_out, [t3_out.size(-1)])

        # gate 输入特征：mean/std + pred_len
        mean = x_.mean(dim=-1)
        std = x_.std(dim=-1)
        pred_len_feat = torch.full((B, 1), L / 1000.0, device=x.device)
        gate_input = torch.cat([mean, std, pred_len_feat], dim=-1)  # [B, 2N+1]

        weights = self.gate_mlp(gate_input)  # [B, 3]
        w1, w2, w3 = weights[:, 0:1, None], weights[:, 1:2, None], weights[:, 2:3, None]

        fused = w1 * t1_out + w2 * t2_out + w3 * t3_out
        fused = self.proj(fused).permute(0, 2, 1)  # [B, L, N]

        return self.norm(self.dropout(fused + x))

class MultiScaleConv(nn.Module):
    def __init__(self, num_vars):
        super().__init__()
        conv_configs = [(3, 1), (5, 2), (7, 3)]
        self.convs = nn.ModuleList([
            nn.Conv1d(num_vars, num_vars, k, padding=(k - 1) * d // 2, dilation=d, bias=False)
            for k, d in conv_configs
        ])
        self.proj = nn.Conv1d(len(conv_configs) * num_vars, num_vars, kernel_size=1)

    def forward(self, x):  # [B, L, N]
        x_ = x.permute(0, 2, 1)
        out = [conv(x_) for conv in self.convs]
        out_cat = torch.cat(out, dim=1)
        out_proj = self.proj(out_cat)
        return out_proj.permute(0, 2, 1)  # [B, L, N]


class HybridTemporalBlock(nn.Module):
    def __init__(self, num_vars, d_model, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len  # 预测长度感知

        self.htm_block = HTMBlock(num_vars, d_model, dropout)
        self.msc_block = MultiScaleConv(num_vars)

        # pred_len-aware gate生成器
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_vars + 1, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # 分支权重：htm、msc、raw
            nn.Softmax(dim=-1)
        )
        self.proj = nn.Sequential(
            nn.Linear(num_vars, num_vars),
            nn.GELU(),
            nn.LayerNorm(num_vars)
        )

    def forward(self, x):  # x: [B, L, N]
        B, L, N = x.shape

        htm_out = self.htm_block(x)  # [B, L, N]
        msc_out = self.msc_block(x)  # [B, L, N]

        # 构造 gate 输入：[x_pool + pred_len]
        x_pool = x.mean(dim=1)  # [B, N]
        pred_len_feat = torch.full((B, 1), self.pred_len / 1000.0, device=x.device)  # 归一化
        gate_input = torch.cat([x_pool, pred_len_feat], dim=-1)  # [B, N+1]

        gate_weights = self.gate_mlp(gate_input)  # [B, 3]
        w_htm = gate_weights[:, 0:1].unsqueeze(-1)
        w_msc = gate_weights[:, 1:2].unsqueeze(-1)
        w_raw = gate_weights[:, 2:3].unsqueeze(-1)

        # 融合输出
        fused = w_htm * htm_out + w_msc * msc_out + w_raw * x  # [B, L, N]
        return self.proj(fused)


class UnifiedRingStarBlock(nn.Module):
    def __init__(self, d_core=64, num_vars=None):
        super(UnifiedRingStarBlock, self).__init__()
        self.d_core = d_core
        self.num_vars = num_vars

        # ============ 初始化各模块 ============
        self.hybrid_temporal = None
        self.variable_router = None

        self.num_centers = None
        self.var_cluster_score = None
        self.var_proj = None
        self.center_proj = None
        self.center_generator = None
        self.center_to_var_proj = None

        self.gate_layer = None
        self.fusion_proj = None
        self.norm = None

    def forward(self, x):  # x: [B, L, N]
        B, L, N = x.shape
        device = x.device
        if self.num_vars is None:
            self.num_vars = N

        # === 初始化 HybridTemporalBlock ===
        if self.hybrid_temporal is None:
            self.hybrid_temporal = HybridTemporalBlock(
                num_vars=N,
                d_model=self.d_core,
                pred_len=L,
                dropout=0.1
            ).to(device)

        x = self.hybrid_temporal(x)  # [B, L, N]

        # === 初始化 VariableRouter ===
        if self.variable_router is None:
            self.variable_router = SparseVariableRouter(num_vars=N).to(device)
        ring_out = self.variable_router(x)  # [B, L, N]

        # === 初始化中心数与参数 ===
        if self.num_centers is None:
            self.num_centers = min(16, max(2, int(math.sqrt(N))))

        if self.var_cluster_score is None:
            self.var_cluster_score = nn.Parameter(torch.randn(N, self.num_centers))  # [N, C]
            self.var_proj = nn.Linear(self.num_centers, self.d_core).to(device)
            self.center_proj = nn.Sequential(
                nn.Linear(self.d_core, self.d_core),
                nn.GELU(),
                nn.Linear(self.d_core, self.d_core)
            ).to(device)
            self.center_to_var_proj = nn.Linear(self.d_core, N).to(device)

        # === 初始化中心向量生成器 ===
        if self.center_generator is None:
            self.center_generator = nn.Linear(N, self.num_centers * self.d_core).to(device)

        # === 动态生成中心 token（每个样本） ===
        summary = x.mean(dim=1)  # [B, N]
        center_tokens = self.center_generator(summary).view(B, self.num_centers, self.d_core)  # [B, C, d]

        # === Soft 聚类（变量 → 中心） ===
        cluster_weight = torch.softmax(self.var_cluster_score.to(device), dim=-1)  # [N, C]

        # === 聚合变量信息 → 中心维度 ===
        x_center = torch.einsum('bln,nc->blc', x, cluster_weight)  # [B, L, C]
        x_center = self.var_proj(x_center)                         # [B, L, d]

        # === 中心 token 处理 ===
        center_tokens = center_tokens.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, C, d]
        center_updated = center_tokens + x_center.unsqueeze(2)           # [B, L, C, d]
        center_updated = self.center_proj(center_updated)                # [B, L, C, d]

        # === 中心 → 变量广播 ===
        fused_center = torch.einsum('blcd,nc->bln', center_updated, cluster_weight)  # [B, L, N]
        #center_to_var = self.center_to_var_proj(fused_center)                        # [B, L, N]
        center_to_var = fused_center  # ✅ fused_center 本身就已经是 [B, L, N]

        # 中心消融
        #center_to_var = x

        # === 融合门控层 ===
        if self.gate_layer is None:
            self.gate_layer = nn.Linear(N * 2, N).to(device)
            self.fusion_proj = nn.Linear(N, N).to(device)
        if self.norm is None:
            self.norm = nn.LayerNorm(N).to(device)

        fusion_input = torch.cat([ring_out, center_to_var], dim=-1)  # [B, L, 2N]
        gate = torch.sigmoid(self.gate_layer(fusion_input))          # [B, L, N]
        fused = gate * ring_out + (1 - gate) * center_to_var         # [B, L, N]
        fused = self.fusion_proj(fused)                              # [B, L, N]                                # 残差连接

        out = self.norm(fused + x)
        return out, None


# =============== 主模型结构 Model ===============
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
                    num_vars=configs.enc_in
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
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]