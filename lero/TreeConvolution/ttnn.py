import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        # q, k, v: (B, T, hidden_size)
        B, T, _ = q.size()
        h, d = self.num_heads, self.head_dim

        # project
        q = self.linear_q(q).view(B, T, h, d).transpose(1, 2)  # (B, h, T, d)
        k = self.linear_k(k).view(B, T, h, d).transpose(1, 2)  # (B, h, T, d)
        v = self.linear_v(v).view(B, T, h, d).transpose(1, 2)  # (B, h, T, d)

        # scaled dot-product
        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, h, T, T)
        if attn_bias is not None:
            scores = scores + attn_bias  # broadcast (B,1,T,T) or (B,h,T,T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        x = torch.matmul(attn, v)  # (B, h, T, d)

        # concat heads
        x = x.transpose(1, 2).contiguous().view(B, T, h * d)
        return self.out_proj(x)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attn_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def forward(self, x, attn_bias=None):
        # Self-attention block
        y = self.self_attn_norm(x)
        y = self.self_attn(y, y, y, attn_bias)
        x = x + self.self_attn_dropout(y)
        # Feed-forward block
        y = self.ffn_norm(x)
        y = self.ffn(y)
        return x + y

class TreeTransformer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 nhead=1, num_layers=4, dim_feedforward=None,
                 dropout_rate=0.01, attention_dropout_rate=0.01):
        super().__init__()
        hidden_size = in_channels
        dim_feedforward = dim_feedforward or 4 * hidden_size

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_size, dim_feedforward,
                         dropout_rate, attention_dropout_rate, nhead)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_size, out_channels)

    def forward(self, flat_data):
        """
        flat_data: (trees, idxes)
          trees: Tensor of shape (B, in_channels, 1 + n_nodes)
          idxes: LongTensor of shape (B, 3*n_nodes, 1)
        Returns:
          Tensor of shape (B, out_channels, T)
        """
        trees, idxes = flat_data
        B, C, T = trees.shape
        attn_bias = build_tree_attention_mask(idxes)

        # prepare input for transformer: (B, T, hidden_size)
        x = trees.permute(0, 2, 1).contiguous()

        for layer in self.encoder_layers:
            x = layer(x, attn_bias)

        x = self.out_proj(x)
        return x[:, 0]


def build_tree_attention_mask(idxes):
    """
    idxes: LongTensor of shape (B, 3*n_nodes, 1)
    Returns:
      attn_bias: Tensor of shape (B, 1, seq_len, seq_len), dtype=torch.float,
                 additive bias (0 for allowed, -inf for masked)
    """
    B, M, _ = idxes.shape
    assert M % 3 == 0, "idxes second dim must be multiple of 3"
    n_nodes = M // 3
    seq_len = n_nodes + 1  # include node-0 placeholder
    triples = idxes.view(B, n_nodes, 3)

    # start with full mask of -inf
    mask = torch.full((B, seq_len, seq_len), float('-inf'), device=idxes.device)
    # allow self-attend
    ids = torch.arange(seq_len, device=idxes.device)
    mask[:, ids, ids] = 0.0

    # allow parent<->child attend
    for b in range(B):
        for i in range(n_nodes):
            p, l, r = triples[b, i].tolist()
            pos = i + 1
            for rel in (p, l, r):
                if rel > 0:
                    mask[b, pos, rel] = 0.0
                    mask[b, rel, pos] = 0.0
    # add head dim
    return mask.unsqueeze(1)  # (B, 1, seq_len, seq_len)
