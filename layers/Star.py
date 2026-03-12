import torch
import torch.nn as nn
import torch.nn.functional as F

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        print("STAR received input shape:", input.shape)

        if input.ndim == 4:
            # 输入是 [B, H, L, D]
            batch_size, n_heads, seq_len, d_series = input.shape
            input = input.reshape(batch_size * n_heads, seq_len, d_series)
            reshape_back = True
        elif input.ndim == 3:
            # 输入是 [B, L, D]
            batch_size, seq_len, d_series = input.shape
            reshape_back = False
        else:
            raise ValueError(f"Unsupported input shape for STAR: {input.shape}")

        # 线性变换
        combined_mean = F.gelu(self.gen1(input))  # [B*H, L, D]
        combined_mean = self.gen2(combined_mean)  # [B*H, L, d_core]

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)  # [B*H, L, d_core]
            ratio = ratio.permute(0, 2, 1)  # [B*H, d_core, L]
            ratio = ratio.reshape(-1, seq_len)  # [B*H * d_core, L]
            indices = torch.multinomial(ratio, 1)  # [B*H * d_core, 1]
            indices = indices.view(input.shape[0], -1, 1).permute(0, 2, 1)  # [B*H, 1, d_core]
            combined_mean = torch.gather(combined_mean, 1, indices)  # [B*H, 1, d_core]
            combined_mean = combined_mean.repeat(1, seq_len, 1)  # [B*H, L, d_core]
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, seq_len, 1)

        # 融合
        combined_mean_cat = torch.cat([input, combined_mean], dim=-1)  # [B*H, L, D + d_core]
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))  # -> [B*H, L, D]
        combined_mean_cat = self.gen4(combined_mean_cat)  # -> [B*H, L, D]
        output = combined_mean_cat

        # reshape 回去
        if reshape_back:
            output = output.reshape(batch_size, n_heads, seq_len, d_series)

        return output, None



