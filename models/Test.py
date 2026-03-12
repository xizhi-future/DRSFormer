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

class ResTCN(nn.Module):
    """
    Lightweight Residual TCN with dynamic dilation schedule.
    - num_vars: channel count (equal to N)
    - Ltcn: number of residual blocks (default 4)
    - kernel: conv kernel size (default 3)
    - ema_alpha: smoothing for running period estimate (default 0.1)
    - min_period, max_period: clamps for estimated period
    Notes:
    - forward 接收 x: [B, N, L] (注意：与你原来 HTMBlock 的 x_ 格式一致)
    - 内部通过自相关（FFT）估计 period（time steps），然后映射到 dilation 列表。
    - 使用 F.conv1d 动态传 dilation 参数，不需要重建 conv 层。
    """

    def __init__(self, num_vars, Ltcn=4, kernel=3, ema_alpha=0.1, min_period=2, max_period=None):
        super().__init__()
        self.num_vars = num_vars
        self.Ltcn = Ltcn
        self.kernel = kernel
        self.ema_alpha = ema_alpha
        self.min_period = min_period
        self.max_period = max_period  # if None will be set at forward based on L//2

        # construct base conv blocks (depthwise + pointwise) - dilation applied dynamically in forward
        self.blocks = nn.ModuleList()
        for _ in range(Ltcn):
            block = nn.ModuleDict({
                'dw': nn.Conv1d(num_vars, num_vars, kernel_size=kernel, padding=(kernel - 1) // 2, dilation=1,
                                groups=num_vars, bias=True),
                'pw1': nn.Conv1d(num_vars, num_vars, kernel_size=1),
                'bn': nn.BatchNorm1d(num_vars),
                'pw2': nn.Conv1d(num_vars, num_vars, kernel_size=1)
            })
            self.blocks.append(block)

        # running period (for stability) - buffer so it's moved with .to(device)
        self.register_buffer('running_period', torch.tensor(float(min_period)))

    def estimate_period_from_batch(self, x):
        # x: [B, N, L] -> collapse B and N by averaging to get stability
        B, N, L = x.shape
        ts = x.mean(dim=(0, 1))  # [L] average over batch and vars
        ts = ts - ts.mean()
        n = L
        # FFT-based autocorrelation (power spectrum -> ifft)
        # compute power spectrum
        f = torch.fft.rfft(ts)
        ps = (f * torch.conj(f)).real
        acf = torch.fft.irfft(ps, n=n)
        acf = acf / (acf[0] + 1e-9)
        # search lag peak in [1, n//2]
        half = max(2, n // 2)
        candidate = acf[1:half + 1]  # shape half
        # if candidate is all nan/flat, fallback to min_period
        if torch.isnan(candidate).all():
            return float(self.min_period)
        # pick largest autocorr lag
        lag = torch.argmax(candidate).item() + 1
        return float(lag)

    def period_to_dilations(self, period, L):
        # clamp
        if self.max_period is None:
            maxp = max(2, L // 2)
        else:
            maxp = self.max_period
        p = int(max(self.min_period, min(int(round(period)), maxp)))
        # map to power-of-two exponents approximate to period magnitude
        e_max = max(0, int(math.floor(math.log2(p))))  # e_max >= 0
        # build list of exponents of length Ltcn aligned to e_max
        if e_max + 1 >= self.Ltcn:
            start = e_max - (self.Ltcn - 1)
            exps = list(range(start, e_max + 1))
        else:
            exps = list(range(0, e_max + 1))
            # pad with last exponent if needed
            while len(exps) < self.Ltcn:
                exps.append(exps[-1])
        dilations = [max(1, 2 ** e) for e in exps]
        # ensure dilations are <= L//2
        dilations = [min(d, max(1, L // 2)) for d in dilations]
        return dilations

    def forward(self, x):
        # x: [B, N, L]
        B, N, L = x.shape
        if self.max_period is None:
            self.max_period = max(2, L // 2)

        # 1) estimate current period from batch
        try:
            period = self.estimate_period_from_batch(x)
        except Exception:
            period = float(self.running_period.item())

        # 2) EMA smooth
        rp = float(self.running_period.item())
        rp = (1.0 - self.ema_alpha) * rp + self.ema_alpha * float(period)
        self.running_period.fill_(rp)

        # 3) map to dilations
        dilations = self.period_to_dilations(rp, L)  # list length Ltcn

        out = x
        for i, block in enumerate(self.blocks):
            d = dilations[i]
            pad = (self.kernel - 1) * d // 2
            # depthwise conv via functional call with dynamic dilation/padding
            out_dw = F.conv1d(out, block['dw'].weight, block['dw'].bias,
                              stride=1, padding=pad, dilation=d, groups=self.num_vars)
            out_pw = block['pw1'](out_dw)
            out_pw = block['bn'](out_pw)
            out_pw = F.gelu(out_pw)
            out_pw = block['pw2'](out_pw)
            # residual
            out = out + out_pw

        return out  # [B, N, L]


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
        # self.t3 = nn.Sequential(
        #     nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars),
        #     nn.GELU(),
        #     nn.Conv1d(num_vars, num_vars, kernel_size=1),
        #     nn.BatchNorm1d(num_vars),
        #     nn.GELU(),
        #     nn.Conv1d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars),
        #     nn.Conv1d(num_vars, num_vars, kernel_size=1)
        # )
        self.t3 = ResTCN(num_vars=num_vars, Ltcn=4, kernel=3, ema_alpha=0.12, min_period=2)

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
            nn.Conv1d(num_vars, num_vars, k, padding=((k - 1) * dil) // 2, dilation=dil, bias=False)
            for k, dil in conv_configs
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


# class UnifiedRingStarBlock(nn.Module):
#     def __init__(self, d_core=64, kernel_size=3, dilation=1, num_vars=None, use_multi_center=True):
#         super(UnifiedRingStarBlock, self).__init__()
#         self.d_core = d_core
#         self.num_vars = num_vars
#         self.hybrid_temporal = None
#         self.norm = None

#     def forward(self, x):  # [B, L, N]
#         B, L, N = x.shape
#         device = x.device

#         if self.hybrid_temporal is None:
#             self.hybrid_temporal = HybridTemporalBlock(
#                 num_vars=N,
#                 d_model=self.d_core,
#                 pred_len=L,
#                 dropout=0.1
#             ).to(device)

#         # 只保留时间建模
#         out = self.hybrid_temporal(x)   # [B, L, N]
#         if self.norm is None:
#             self.norm = nn.LayerNorm(N).to(device)
#         return self.norm(out + x), None

class UnifiedRingStarBlock(nn.Module):
    def __init__(self, d_core=64, num_vars=None, suppress_outliers=True):
        super(UnifiedRingStarBlock, self).__init__()
        self.d_core = d_core
        self.num_vars = num_vars
        self.suppress_outliers = suppress_outliers  # 控制异常值抑制

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

        # ============ 星型中心聚合：添加异常值抑制 ============
        # 对输入x进行轻量级Winsorize处理，减少异常值影响
        if self.suppress_outliers:
            # 计算每个变量的5%和95%分位数（保留95%的数据）
            with torch.no_grad():
                # sort along time dimension
                x_sorted, _ = torch.sort(x, dim=1)  # [B, L, N]
                # guard indices
                low_idx = max(0, int(L * 0.05))
                high_idx = min(L - 1, int(L * 0.95))
                lower_bounds = x_sorted[:, low_idx, :]  # [B, N]
                upper_bounds = x_sorted[:, high_idx, :]  # [B, N]

            # 用逐元素的 min/max 来实现 per-element clamp，避免向 clamp 传 tensor 导致不同 torch 版本报错
            lower = lower_bounds.unsqueeze(1)  # [B,1,N] -> broadcast to [B,L,N]
            upper = upper_bounds.unsqueeze(1)  # [B,1,N]

            # ensure lower <= upper elementwise (数值稳定性)
            # 如果某些情况下 lower > upper（极端/数值误差），用 min/max 修正
            tmp_min = torch.min(lower, upper)
            tmp_max = torch.max(lower, upper)
            lower = tmp_min
            upper = tmp_max

            # 等价于 clamp(x, min=lower, max=upper) 的分步实现
            x_clamped_low = torch.max(x, lower)  # 保证 >= lower
            x_clipped = torch.min(x_clamped_low, upper)  # 保证 <= upper

            # 混合原始和截断数据（90%截断 + 10%原始，平滑过渡）
            x_mixed = 0.9 * x_clipped + 0.1 * x
            x_center = torch.einsum('bln,nc->blc', x_mixed, cluster_weight)  # [B, L, C]
        else:
            x_center = torch.einsum('bln,nc->blc', x, cluster_weight)  # [B, L, C]

        x_center = self.var_proj(x_center)  # [B, L, d]

        # === 中心 token 处理 ===
        center_tokens = center_tokens.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, C, d]
        center_updated = center_tokens + x_center.unsqueeze(2)  # [B, L, C, d]
        center_updated = self.center_proj(center_updated)  # [B, L, C, d]

        # ============ 广播中心聚合：添加异常值抑制 ============
        fused_center = torch.einsum('blcd,nc->bln', center_updated, cluster_weight)  # [B, L, N]

        if self.suppress_outliers:
            # 基于运行统计的异常值抑制
            if not hasattr(self, 'running_mean'):
                self.register_buffer('running_mean', torch.zeros(1, 1, N, device=device))
                self.register_buffer('running_std', torch.ones(1, 1, N, device=device))

            # 更新运行统计（指数移动平均）
            with torch.no_grad():
                batch_mean = fused_center.mean(dim=(0, 1), keepdim=True)
                batch_std = fused_center.std(dim=(0, 1), keepdim=True) + 1e-5
                self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
                self.running_std = 0.9 * self.running_std + 0.1 * batch_std

            # 计算z-score并创建高斯抑制门控
            z_scores = (fused_center - self.running_mean) / self.running_std  # [B, L, N]
            suppression_gate = torch.exp(-0.5 * (z_scores / 3.0) ** 2)  # 高斯门控，|z|>3时权重<0.61

            # 应用门控，保持残差连接
            fused_center_suppressed = fused_center * suppression_gate
            center_to_var = 0.8 * fused_center_suppressed + 0.2 * fused_center  # 保留部分原始信号
        else:
            center_to_var = fused_center  # ✅ fused_center 本身就已经是 [B, L, N]
        # center_to_var = x

        # === 融合门控层 ===
        #         if self.gate_layer is None:
        #             self.gate_layer = nn.Linear(N * 2, N).to(device)
        #             self.fusion_proj = nn.Linear(N, N).to(device)
        #         if self.norm is None:
        #             self.norm = nn.LayerNorm(N).to(device)

        #         fusion_input = torch.cat([ring_out, center_to_var], dim=-1)  # [B, L, 2N]
        #         gate = torch.sigmoid(self.gate_layer(fusion_input))  # [B, L, N]
        #         fused = gate * ring_out + (1 - gate) * center_to_var  # [B, L, N]
        #         fused = self.fusion_proj(fused)  # [B, L, N]  # 残差连接

        #         out = self.norm(fused + x)

        #         if self.training and torch.rand(1).item() < 0.01:
        #             print("Gate mean:", gate.mean().item())
        #             print("Gate std:", gate.std().item())

        # === 融合层（升级版） ===
        if self.gate_layer is None:
            self.gate_layer = nn.Linear(N * 3, N).to(device)
            self.fusion_proj = nn.Linear(N, N).to(device)
        if self.norm is None:
            self.norm = nn.LayerNorm(N).to(device)
        interaction = ring_out * center_to_var
        fusion_input = torch.cat([
            ring_out,
            center_to_var,
            interaction
        ], dim=-1)
        gate = torch.sigmoid(self.gate_layer(fusion_input))
        fused = gate * ring_out + (1 - gate) * center_to_var
        fused = self.fusion_proj(fused)
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