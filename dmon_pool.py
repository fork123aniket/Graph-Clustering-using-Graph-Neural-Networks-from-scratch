from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.dense.mincut_pool import _rank3_trace

EPS = 1e-15


class DMoNPooling(torch.nn.Module):
    def __init__(self, channels: Union[int, List[int]], k: int,
                 dropout: float = 0.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        from torch_geometric.nn.models.mlp import MLP
        self.mlp = MLP(channels + [k], act='selu', norm=None)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = F.selu(torch.matmul(s.transpose(1, 2), x))
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        degrees = torch.einsum('ijk->ik', adj).transpose(0, 1)
        m = torch.einsum('ij->', degrees)

        ca = torch.matmul(s.transpose(1, 2), degrees)
        cb = torch.matmul(degrees.transpose(0, 1), s)

        normalizer = torch.matmul(ca, cb) / 2 / m
        decompose = out_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = torch.mean(spectral_loss)

        # Orthogonality regularization:
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # Cluster loss:
        cluster_loss = torch.norm(torch.einsum(
            'ijk->ij', ss)) / adj.size(1) * torch.norm(i_s) - 1

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mlp.in_channels}, '
                f'num_clusters={self.mlp.out_channels})')