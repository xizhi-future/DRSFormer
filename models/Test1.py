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

        # self.topk = max(16, int(math.log2(num_vars) * 3))
        self.topk = min(num_vars, max(16, int(math.log2(num_vars) * 3)))
        #         self.topk = 1
        self.static_topk = max(32, int(math.sqrt(num_vars) * 2))
        # self.static_topk = 0

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
        """
        Build a robust static mask for variable candidates.
        Minimal, low-intrusion change: winsorize(sim) -> symmetrize -> topk.
        """
        with torch.no_grad():
            B, L, N = x.shape
            # 原始统计（保持你现有逻辑）
            x_avg = x.mean(dim=0)  # [L, N] (mean over batch)
            x_std = x.std(dim=0) + 1e-5  # [L, N]
            normed = x_avg / x_std  # [L, N]
            # 原始相似度矩阵
            sim = torch.matmul(normed.T, normed)  # [N, N]
            # -----------------------
            # 1) winsorize / 去极值：对 sim 的全局分布做上下限截断
            # -----------------------
            # 可调超参：winsor_q = 0.05 表示去掉两端各 5% 的极端值
            winsor_q = getattr(self, 'winsor_q', 0.05)
            if winsor_q is not None and winsor_q > 0.0:
                flat = sim.flatten()
                # quantile 计算（返回 scalar tensor）
                low_q = torch.quantile(flat, winsor_q)
                high_q = torch.quantile(flat, 1.0 - winsor_q)
                # clamp（去掉极端高低值）
                sim = sim.clamp(min=low_q.item(), max=high_q.item())
            # -----------------------
            # 2) 对称化 & 去自环 & 非负化
            # -----------------------
            sim = 0.5 * (sim + sim.t())  # 强制对称，避免数值漂移导致不对称关系
            sim.fill_diagonal_(0.0)  # 排除自连接
            sim = torch.clamp(sim, min=0.0)
            # -----------------------
            # 3) top-k -> mask
            # -----------------------
            _, topk_idx = torch.topk(sim, self.static_topk, dim=-1)
            mask = torch.zeros(N, N, device=x.device)
            mask.scatter_(1, topk_idx, 1.0)
            self.static_mask = mask.bool()

    def forward(self, x):  # [B, L, N]
        B, L, N = x.shape
        if self.static_mask is None:
            self.build_static_mask(x)
        # print("static_topk:", self.static_topk)
        # print("topk:", self.topk)
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

class HTMBlock(nn.Module):
    def __init__(self, num_vars, d_model, dropout=0.1, dilations=(1, 2, 4), t3_blend_beta=0.8):
        """
        Minimal-invasion HTMBlock with robust, multi-scale, period-aware modulation for t3.
        - num_vars: channel count (N)
        - dilations: tuple of dilation rates for the auxiliary t3 branches (kept small, e.g. (1,2,4))
        - t3_blend_beta: fixed blending factor between combined t3 output and identity (0..1)
        """
        super(HTMBlock, self).__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.dilations = tuple(dilations)
        self.num_t3_branches = 1 + len(self.dilations)  # base + aux
        self.t3_blend_beta = float(t3_blend_beta)

        # branch1: trend smoothing
        self.t1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=7, stride=1, padding=3),
            nn.Conv1d(num_vars, num_vars, kernel_size=1),
            nn.BatchNorm1d(num_vars)
        )

        # branch2: grouped depthwise
        self.t2 = nn.Sequential(
            nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm1d(num_vars),
            nn.GELU(),
            nn.Conv1d(num_vars, num_vars, kernel_size=1),
        )

        # base t3: keep your original effective block intact
        self.t3_base = nn.Sequential(
            nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars),
            nn.GELU(),
            nn.Conv1d(num_vars, num_vars, kernel_size=1),
            nn.BatchNorm1d(num_vars),
            nn.GELU(),
            nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars),
            nn.Conv1d(num_vars, num_vars, kernel_size=1)
        )

        # auxiliary dilated branches (depthwise conv + pw conv)
        self.t3_aux = nn.ModuleList()
        for d in self.dilations:
            pad = (3 - 1) * d // 2
            block = nn.Sequential(
                nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=pad, dilation=d, groups=num_vars, bias=True),
                nn.GELU(),
                nn.Conv1d(num_vars, num_vars, kernel_size=1),
                nn.BatchNorm1d(num_vars)
            )
            self.t3_aux.append(block)

        # period-aware gate for combining t3 branches (base + aux)
        self.t3_period_gate = nn.Sequential(
            nn.Linear(2 * num_vars, 64),
            nn.GELU(),
            nn.Linear(64, self.num_t3_branches)  # logits for softmax
        )
        # initialize bias to favor base branch (index 0)
        with torch.no_grad():
            # bias shape [num_t3_branches], set bias so softmax favors base
            b = torch.zeros(self.num_t3_branches)
            b[0] = 2.0
            if self.num_t3_branches > 1:
                b[1:] = -1.0
            self.t3_period_gate[-1].bias.copy_(b)

        # original gate that fuses t1,t2,t3 (kept unchanged)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * num_vars + 1, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        self.proj = nn.Conv1d(num_vars, num_vars, kernel_size=1)
        self.norm = nn.LayerNorm(num_vars)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, L, N]
        B, L, N = x.shape
        x_ = x.permute(0, 2, 1)  # [B, N, L]

        # branch outputs
        t1_out = self.t1(x_)  # [B, N, L]
        t2_out = self.t2(x_)  # [B, N, L]
        t3_base_out = self.t3_base(x_)  # [B, N, L]

        # compute aux t3 branch outputs
        t3_aux_outs = []
        for blk in self.t3_aux:
            t3_aux_outs.append(blk(x_))  # list of [B,N,L]

        # ---------- period feature (lightweight & stable) ----------
        mean_feat = x_.mean(dim=-1)  # [B, N]
        std_feat = x_.std(dim=-1)  # [B, N]
        period_feat = torch.cat([mean_feat, std_feat], dim=-1)  # [B, 2N]

        # ---------- t3 branch soft combination ----------
        # produce soft weights per branch from period features
        t3_logits = self.t3_period_gate(period_feat)  # [B, num_t3_branches]
        t3_weights = F.softmax(t3_logits, dim=-1).unsqueeze(-1).unsqueeze(
            -1)  # [B, num_br, 1,1] after unsqueeze pattern

        # construct stacked branches tensor [B, num_br, N, L]
        # base branch first
        branches = [t3_base_out] + t3_aux_outs  # list length num_br
        stacked = torch.stack(branches, dim=1)  # [B, num_br, N, L]

        # apply weights and sum
        weighted = (t3_weights * stacked).sum(dim=1)  # [B, N, L]

        # blend with identity to preserve stability
        beta = self.t3_blend_beta
        t3_out = beta * weighted + (1.0 - beta) * x_  # [B, N, L]

        # ---------- branch normalization ----------
        t1_out = F.layer_norm(t1_out, [t1_out.size(-1)])
        t2_out = F.layer_norm(t2_out, [t2_out.size(-1)])
        t3_out = F.layer_norm(t3_out, [t3_out.size(-1)])

        # ---------- fusion gate (original) ----------
        pred_len_feat = torch.full((B, 1), L / 1000.0, device=x.device)
        gate_input = torch.cat([mean_feat, std_feat, pred_len_feat], dim=-1)  # [B, 2N+1]
        weights = self.gate_mlp(gate_input)  # [B, 3]
        w1 = weights[:, 0:1].unsqueeze(-1)
        w2 = weights[:, 1:2].unsqueeze(-1)
        w3 = weights[:, 2:3].unsqueeze(-1)

        fused = w1 * t1_out + w2 * t2_out + w3 * t3_out  # [B, N, L]

        fused = self.proj(fused).permute(0, 2, 1)  # -> [B, L, N]
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
    def __init__(self, d_core=64, kernel_size=3, dilation=1, num_vars=None, use_multi_center=True):
        super(UnifiedRingStarBlock, self).__init__()
        self.d_core = d_core
        self.num_vars = num_vars
        self.hybrid_temporal = None
        self.norm = None

    def forward(self, x):  # [B, L, N]
        B, L, N = x.shape
        device = x.device

        if self.hybrid_temporal is None:
            self.hybrid_temporal = HybridTemporalBlock(
                num_vars=N,
                d_model=self.d_core,
                pred_len=L,
                dropout=0.1
            ).to(device)

        # 只保留时间建模
        out = self.hybrid_temporal(x)   # [B, L, N]
        if self.norm is None:
            self.norm = nn.LayerNorm(N).to(device)
        return self.norm(out + x), None

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