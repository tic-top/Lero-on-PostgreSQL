import torch
import torch.nn as nn

class TreeTransformer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 nhead=1, num_layers=4, dim_feedforward=None):
        super(TreeTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dim_feedforward = dim_feedforward or 4 * in_channels

        # TransformerEncoder: 每层 d_model=in_channels，nhead 头
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False  # 我们用 (seq, batch, dim) 这个格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        outs = []
        for b in range(B):
            # attn_mask[b]: (T, T)，True 表示屏蔽
            out_b = self.encoder(src[:, b : b + 1, :], mask=attn_masks[b])
            # out_b: (T, 1, C) -> (T, C)
            outs.append(out_b.squeeze(1))
        # 拼回 (T, B, C)
        out = torch.stack(outs, dim=1)

        # 4) 投影到 out_channels
        #    out: (T, B, C) -> (B, T, C)
        out = out.permute(1, 0, 2)
        # 再做线性层 (B, T, C_in) -> (B, T, out_channels)
        out = self.out_proj(out)
        # 最终转成 (B, out_channels, T)
        results = out.permute(0, 2, 1)

        return results[:,:,0]
    

def build_tree_attention_mask(idxes, seq_len):
    """
    idxes: LongTensor of shape (B, M, 1), 每组三元组按[ parent, left, right ] 编号，叶子处会有 0。
    seq_len: int, 序列总长度 T（flat_trees 的第二维）
    返回: attn_mask of shape (B, T, T), dtype=torch.bool，True 表示“屏蔽”。
    """
    B, M1, _ = idxes.shape
    # 先把 M1 = 3 * n_nodes 展平回 (B, n_nodes, 3)
    n_nodes = M1 // 3
    triples = idxes.view(B, n_nodes, 3)  # (B, n_nodes, 3)

    # 初始化允许 attention 的关系矩阵（默认只允许自身 attend 自身）
    A = torch.eye(seq_len, dtype=torch.bool, device=idxes.device).unsqueeze(0).expand(B, -1, -1)

    for b in range(B):
        for i in range(n_nodes):
            p, l, r = triples[b, i].tolist()
            # 这里 i 对应的是序号 i+1，因为 0 号是那个全零占位
            pos = i + 1  
            # 自己肯定能 attend 自己（上面已 init），下面补 parent/child 关系
            if p > 0:
                A[b, pos, p] = True
                A[b, p, pos] = True
            if l > 0:
                A[b, pos, l] = True
                A[b, l, pos] = True
            if r > 0:
                A[b, pos, r] = True
                A[b, r, pos] = True

    # Transformer 需要的 attn_mask: True 表示“屏蔽”，False 表示“可 attend”
    attn_mask = ~A  # (B, T, T)
    return attn_mask
