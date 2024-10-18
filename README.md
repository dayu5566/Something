import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义线性注意力机制
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(LinearAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)

        attention_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.embed_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        context = torch.bmm(attention_weights, v)
        output = self.output_projection(context)
        return output


# 定义相对位置编码
class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


# 定义自适应的卷积块，添加残差连接
class AdaptiveTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(AdaptiveTemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = self.downsample(x) if self.downsample is not None else x
        if out.size(2) != res.size(2):
            res = torch.nn.functional.pad(res, (0, out.size(2) - res.size(2)))

        return self.relu(out + res)


# 定义自适应的TCN网络
class AdaptiveTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(AdaptiveTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [AdaptiveTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
                                            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
