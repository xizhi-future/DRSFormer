import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


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
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        # 在训练模式下，使用随机池化（stochastic pooling）选择特征；在非训练模式下，使用加权平均
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        # 将输入和处理后的特征连接起来，通过另外两个线性层和GELU激活函数生成最终输出
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat
        # 返回输出和None
        return output, None


class Model(nn.Module):
    '''
        数据嵌入：使用 DataEmbedding_inverted 将输入数据转换为适合模型处理的嵌入表示。
        编码器：使用 Encoder 和 EncoderLayer 构建Transformer编码器，通过多层堆叠实现特征提取。
        随机池化：在 STAR 模块中实现随机池化和加权平均，增强模型的表达能力。
        归一化：可选的归一化步骤，用于标准化输入和输出数据，提高模型的稳定性。
        投影层：将编码器的输出映射到预测长度，生成最终的预测结果。
        前向传播：通过 forward 方法定义模型的前向传播过程，包括数据预处理、编码器处理和输出生成。
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        # 初始化数据嵌入层和归一化标志
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        # Encoder
        # 初始化编码器，使用多个 EncoderLayer 堆叠
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )
        # Decoder
        # 初始化投影层，将编码器的输出映射到预测长度
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # 如果使用归一化，对输入数据进行标准化处理
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 通过嵌入层和编码器处理输入数据，然后通过投影层生成预测输出
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        # 如果使用归一化，对输出数据进行反标准化处理
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    # 定义前向传播函数，调用 forecast 方法生成预测输出，并返回最后一部分作为最终预测结果
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
