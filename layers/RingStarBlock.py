import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 新的
class SparseVariableRouter(nn.Module):
    def __init__(self, num_vars, hidden_dim=16, topk=8, temp=1.0, static_topk=30):
        super(SparseVariableRouter, self).__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        self.topk = max(16, int(math.log2(num_vars) * 3))
        self.static_topk = max(32, int(math.sqrt(num_vars) * 2))
        self.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)

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

        x_var = x.permute(0, 2, 1)  # [B, N, L]
        static_emb = self.var_embed.expand(B, -1, -1)

        mean = x_var.mean(dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x_var, dim=-1, keepdim=True) + 1e-5)
        time_feat = torch.cat([mean, std], dim=-1)
        time_emb = self.temporal_proj(time_feat)

        var_fused = torch.cat([static_emb, time_emb], dim=-1)
        importance = self.importance_gate(var_fused).squeeze(-1).unsqueeze(1)

        Q = self.query_proj(var_fused)
        K = self.key_proj(var_fused)
        sim_matrix = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        mask = self.static_mask.unsqueeze(0).expand(B, -1, -1)
        sim_matrix = sim_matrix.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(sim_matrix, dim=-1)  # ✅ 去除 Gumbel + topk

        # 全连接 attention，不提取 topk
        routed = torch.bmm(attn_weights, x_var)  # [B, N, L]
        routed_out = routed.permute(0, 2, 1)  # [B, L, N]

        return importance * routed_out + (1 - importance) * x

class RingStarBlock(nn.Module):
    def __init__(self, d_core=64, kernel_size=3, dilation=1, num_vars=None, use_multi_center=True):
        super(RingStarBlock, self).__init__()
        self.d_core = d_core
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_vars = num_vars
        self.use_multi_center = use_multi_center

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

#         # === 初始化 HybridTemporalBlock ===
#         if self.hybrid_temporal is None:
#             self.hybrid_temporal = HybridTemporalBlock(
#                 num_vars=N,
#                 d_model=self.d_core,
#                 pred_len=L,
#                 dropout=0.1
#             ).to(device)

#         x = self.hybrid_temporal(x)  # [B, L, N]

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

        # === 融合门控层 ===
        if self.gate_layer is None:
            self.gate_layer = nn.Linear(N * 2, N).to(device)
            self.fusion_proj = nn.Linear(N, N).to(device)
        if self.norm is None:
            self.norm = nn.LayerNorm(N).to(device)

        fusion_input = torch.cat([ring_out, center_to_var], dim=-1)  # [B, L, 2N]
        gate = torch.sigmoid(self.gate_layer(fusion_input))          # [B, L, N]
        fused = gate * ring_out + (1 - gate) * center_to_var         # [B, L, N]
        fused = self.fusion_proj(fused)                              # [B, L, N]
        # 残差连接
        out = self.norm(fused + x)
        return out, None

