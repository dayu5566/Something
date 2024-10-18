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
