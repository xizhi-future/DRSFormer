import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class TemporalBlock(nn.Module):
    """
    功能：1、时间序列特征提取：使用卷积层提取时间序列的局部特征，通过扩展感受野(dialation)捕捉长距离依赖；
         2、深层结构：通过堆叠多个TemporalBlock,可以逐步学习更高层次的时间序列特征；
         3、正则化：Dropout减少过拟合风险；
         4、非线性建模：ReLU激活函数引入非线性能力。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        参数：
        n_inputs:输入通道数；
        n_outputs:输出通道数；
        kernel_size:卷积核的大小;
        stride:卷积操作的步长，通常为1;
        dilation:空洞卷积的扩展因子，用于增加感受野；
        padding:用于对序列数据进行填充，使输出与输入长度一致；
        dropout:随机丢弃神经元的比例，用于正则化。

        作用：定义了两个卷积层conv1和conv2,每层后接Dropout和ReLU激活函数。
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        输入张量x的维度是(batch_size,n_inputs,seq_len),表示批量时间序列数据；
        batch_size:批量大小；
        n_inputs:每个时间步的特征数量；
        seq_len:时间步的长度。

        第一步：通过第一个卷积层conv1提取特征；应用Dropout防止过拟合；使用ReLU激活函数添加非线性；
        第二步：输出再次通过第二个卷积层conv2;应用Dropout和ReLU
        最后，将处理后的序列返回。
        """
        out = self.relu(self.dropout1(self.conv1(x)))
        out = self.relu(self.dropout2(self.conv2(out)))
        return out


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cuda"):
        super(TCN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        kernel_size = 3  # 定义卷积核大小
        self.tcn_layers = nn.ModuleList()
        num_channels = [self.input_size] + [self.hidden_size] * (self.num_layers - 1) + [self.hidden_size]

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i + 1]
            padding = (kernel_size - 1) * dilation // 2
            self.tcn_layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=padding,
                              dropout=0.2))

    # def forward(self, x_enc):
    #     batch_size, seq_len = x_enc.shape[0], x_enc.shape[1]  # batch_size=32, seq_len=30, hidden_size=64
    #     x_enc = x_enc.permute(0, 2, 1)  # 变换为 [batch_size, input_size, seq_len]
    #     for layer in self.tcn_layers:
    #         x_enc = layer(x_enc)
    #     x_enc = x_enc.permute(0, 2, 1)  # 变回 [batch_size, seq_len, hidden_size]
    #     return x_enc  # torch.Size([batch_size, seq_len, hidden_size])
    def forward(self, x_enc):
        # 不再 permute，输入就是 [B, C, T]
        for layer in self.tcn_layers:
            x_enc = layer(x_enc)
        return x_enc  # 保持 [B, C, T]


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # 统一嵌入
        self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                configs.dropout)
        # 时间维度建模：TCN
        self.tcn = TCN(input_size=configs.d_model, hidden_size=configs.d_model, num_layers=3,
                       batch_size=configs.batch_size)
        # 变量维度建模：自注意力机制
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 门控融合
        self.gate = nn.Linear(configs.d_model * 2, configs.d_model)

        # 投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # 统一嵌入
        enc_out = self.embedding(x_enc, x_mark_enc)
        #print("enc_out.shape: ", enc_out.shape)     # [B, T, D]

        # 时间维度建模：TCN
        tcn_input = enc_out.permute(0, 2, 1)  # [B, D, T]
        #print("tcn_input.shape: ", tcn_input.shape)
        tcn_out = self.tcn(tcn_input).permute(0, 2, 1)  # 输出再转回 [B, T, D]
        #print("tcn_out.shape: ", tcn_out.shape)
        # 变量维度建模：自注意力机制
        attn_out, attns = self.encoder(enc_out, attn_mask=None)

        # 门控融合
        combined_out = torch.cat((tcn_out, attn_out), dim=-1)
        combined_out = self.gate(combined_out)

        dec_out = self.projection(combined_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None