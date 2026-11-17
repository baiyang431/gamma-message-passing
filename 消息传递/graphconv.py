"""用 PyTorch 张量实现 DGL norm='both' 的同构图卷积。"""

import torch
from torch import nn


def normalize(edges, num_nodes, dtype, self_loops):
    if self_loops:
        # 先移除已有自环，确保每个节点恰好添加一个自环。
        edges = edges[:, edges[0] != edges[1]]
        nodes = torch.arange(num_nodes, device=edges.device)
        edges = torch.cat((edges, torch.stack((nodes, nodes))), dim=1)
    src, dst = edges
    out_degree = torch.bincount(src, minlength=num_nodes).to(dtype).clamp_min(1)
    in_degree = torch.bincount(dst, minlength=num_nodes).to(dtype).clamp_min(1)
    norm = out_degree[src].rsqrt() * in_degree[dst].rsqrt()
    return edges, norm


class GraphConv(nn.Module):
    """有向多重图卷积；self_loops=True 时将自环统一为每节点一个。"""

    def __init__(self, in_feats, out_feats, self_loops=True):
        super().__init__()
        self.self_loops = self_loops
        self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        self.bias = nn.Parameter(torch.zeros(out_feats))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edges):
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError("x 必须为 [N, in_feats]")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
            raise ValueError("edges 必须为 int64 张量，形状 [2, E]")
        if edges.device != x.device:
            raise ValueError("edges 和 x 必须位于同一设备")
        if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
            raise ValueError("边端点超出节点索引范围")
        if not torch.isfinite(x).all():
            raise ValueError("节点特征包含 NaN 或无穷值")
        edges, norm = normalize(edges, len(x), x.dtype, self.self_loops)
        src, dst = edges
        hidden = x @ self.weight
        messages = hidden[src] * norm.unsqueeze(1)
        aggregated = hidden.new_zeros((len(x), self.weight.shape[1]))
        aggregated.index_add_(0, dst, messages)
        return aggregated + self.bias
