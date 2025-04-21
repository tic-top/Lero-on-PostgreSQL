import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class TreeTransformer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 nhead=1, num_layers=4, dim_feedforward=None):
        super(TreeTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dim_feedforward = dim_feedforward or 4 * in_channels

        # TransformerEncoder: 每层 d_model=in_channels，nhead 头
        self.encoder_layers = [EncoderLayer(in_channels,2*in_channels, 0.01, 0.01, nhead)
                    for _ in range(num_layers)]

        # 最后把 d_model 映射到 out_channels
        self.out_proj = nn.Linear(in_channels, out_channels)

    def forward(self, flat_data):
        """
        flat_data: tuple of
          trees: Tensor of shape (B, in_channels, T)
          idxes: LongTensor of shape (B, 3*n_nodes, 1)
        返回:
          (results, orig_idxes), 
          results.shape == (B, out_channels, T)
        """
        trees, idxes = flat_data
        orig_idxes = idxes
        B, C, T = trees.shape

        # 1) 构造 attention_mask，形状 (B, T, T)
        attn_masks = build_tree_attention_mask(idxes, seq_len=T)  # 参见前面定义

        # 2) 准备序列 (seq_len, batch, feature)
        #    把 (B, C, T) -> (T, B, C)
        src = trees.permute(2, 0, 1)

        # 3) 逐样本跑 encoder，利用各自的 attn_mask
        for enc in self.encoder_layers:
            src = enc(src, attn_masks)
        src = src.permute(1, 0, 2)
        src = self.out_proj(src)
        results = src.permute(0, 2, 1)

        return results[:,:,0]
    

def build_tree_attention_mask(idxes, seq_len):
    """
    idxes: LongTensor of shape (B, M, 1), M 为 3 * n_nodes，每组 3 个值按 [parent, left, right] 编号，
           节点 0 是占位（根之前），真实节点位置是 1..n_nodes。
    返回: attn_mask of shape (B, n_nodes+1, n_nodes+1), dtype=torch.bool，
          True 表示“屏蔽”，False 表示“可 attend”。
    """
    B, M, _ = idxes.shape
    assert M % 3 == 0, "idxes 的第二维必须是 3 的倍数"
    n_nodes = M // 3
    triples = idxes.view(B, n_nodes, 3)   # (B, n_nodes, 3)
    seq_len = n_nodes + 1                 # 包含 0 号占位

    # 初始化全屏蔽
    mask = -100 * torch.ones(B, seq_len, seq_len, device=idxes.device)
    # 允许每个 token attend 自己
    ids = torch.arange(seq_len, device=idxes.device)
    mask[:, ids, ids] = 0

    # 遍历 parent/left/right，双方互通
    for b in range(B):
        for i in range(n_nodes):
            p, l, r = triples[b, i].tolist()
            pos = i + 1  # 真正的序号
            for rel in (p, l, r):
                if rel > 0:
                    mask[b, pos, rel] = 0
                    mask[b, rel, pos] = 0

    return mask
