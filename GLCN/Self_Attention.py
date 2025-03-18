import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
import sys

sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
from GTN.inits import glorot

cfg = get_cfg()
print(torch.cuda.device_count())
device = torch.device(cfg.cuda if torch.cuda.is_available() else "cpu")
print(device)


def sample_adjacency_matrix(weight_matrix):
    # weight_matrix는 n x n 텐서이며, 각 원소는 연결 확률을 나타냅니다.
    # 0과 1 사이의 uniform random matrix를 생성합니다.
    random_matrix = torch.rand(weight_matrix.size()).to(device)
    adjacency_matrix = (random_matrix < weight_matrix).int()

    return adjacency_matrix


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5,
                   mini_batch: bool = False) -> Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()
    logits_for_improvements = y_soft
    if hard:
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret, logits_for_improvements


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()


class GAT(nn.Module):
    def __init__(self, feature_size, graph_embedding_size):
        super(GAT, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.Ws = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
        glorot(self.Ws)
        self.a = nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, Wq, Wv, mini_batch=False):
        if mini_batch == False:
            Wh1 = Wq
            Wh2 = Wv
            Wh1 = torch.matmul(Wh1, self.a[:self.graph_embedding_size, :])
            Wh2 = torch.matmul(Wh2, self.a[self.graph_embedding_size:, :])
            e = Wh1 + Wh2.T
        else:
            Wh1 = Wq
            Wh2 = Wv
            Wh1 = torch.matmul(Wh1, self.a[:self.graph_embedding_size, :])
            Wh2 = torch.matmul(Wh2, self.a[self.graph_embedding_size:, :])
            e = Wh1 + torch.transpose(Wh2, 1, 2)
        return F.leaky_relu(e, negative_slope=cfg.negativeslope)

    def forward(self, A, X, dead_masking=False, mini_batch=False, dense=False):
        if mini_batch == False:
            E = A.to(device)
            num_nodes = X.shape[0]
            if dense == False:
                E = torch.sparse_coo_tensor(E.clone().detach(),
                                            torch.ones(torch.tensor(E.clone().detach()).shape[1]).to(device),
                                            (num_nodes, num_nodes)).long().to(device).to_dense()
            else:
                E = A
            Wh = X @ self.Ws
            a = self._prepare_attentional_mechanism_input(Wh, Wh)
            zero_vec = -9e15 * torch.ones_like(E)
            a = torch.where(E > 0, a, zero_vec)
            a = F.softmax(a, dim=1)
            H = torch.matmul(a, Wh)
        else:
            batch_size, num_nodes, feature_size = X.shape
            Wh = X @ self.Ws  # [batch_size, num_nodes, graph_embedding_size]
            if dense == False:
                E = torch.stack([
                    torch.sparse_coo_tensor(
                        torch.tensor(A[b], dtype=torch.long).to(device),
                        torch.ones(torch.tensor(A[b]).shape[1]).to(device),
                        (num_nodes, num_nodes)
                    ).to_dense()
                    for b in range(batch_size)
                ], dim=0)
            else:
                E = A

            e = self._prepare_attentional_mechanism_input(Wh, Wh, mini_batch=mini_batch)

            zero_vec = -9e15 * torch.ones_like(E)

            attention = torch.where(E > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            H = torch.matmul(attention, Wh)
        return H


class GLCN(nn.Module):
    def __init__(self, feature_size,
                 graph_embedding_size,
                 feature_obs_size):
        super(GLCN, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.feature_obs_size = feature_obs_size
        self.a_link = nn.Parameter(torch.empty(size=(self.feature_obs_size, 1)))
        nn.init.xavier_uniform_(self.a_link.data, gain=1.414)

    def forward(self, h, mini_batch=False):
        h = h.detach()
        if mini_batch == False:
            h = h[:, :self.feature_obs_size]
            h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
            h = h.squeeze(2)
            A, logits_for_improvements = gumbel_sigmoid(h, tau=float(os.environ.get("gumbel_tau", 1.0)), hard=True,
                                                        threshold=0.5)
            A = A.unsqueeze(0)
            batch_size, num_nodes = A.shape[0], A.shape[1]
            mask = torch.ones_like(A)

            indices = torch.arange(num_nodes, device=A.device)
            mask[:, indices, indices] = 0.0

            batch_size, n, n = logits_for_improvements.shape
            mask = ~torch.eye(n, dtype=torch.bool, device=logits_for_improvements.device).unsqueeze(0).expand(
                batch_size, -1, -1)
            # A가 1인 위치의 확률과 0인 위치의 (1-확률) 계산
            # 사칙연산만 사용하여 그래디언트가 잘 흐르도록 함
            selected_probs = A * logits_for_improvements + (1 - A) * (1 - logits_for_improvements)

            # 비대각 요소만 선택
            masked_probs = selected_probs.masked_select(mask).view(batch_size, -1)

            # 로그 도메인에서 합산 (수치적 안정성을 위해)
            log_probs = torch.log(masked_probs + 1e-8)  # 수치적 안정성을 위한 작은 값 추가
            summed_values = log_probs.sum(dim=1)
            A = A.squeeze(0)
            A = A - torch.diag(torch.diag(A))
            I = torch.eye(h.shape[1]).to(device)
            A = A + I
            return A, summed_values
        else:
            h = h[:, :, :self.feature_obs_size]
            h = torch.einsum("bijk,kl->bijl", torch.abs(h.unsqueeze(2) - h.unsqueeze(1)), self.a_link)
            h = h.squeeze(3)

            A, logits_for_improvements = gumbel_sigmoid(h, tau=float(os.environ.get("gumbel_tau", 1.0)), hard=True,
                                                        threshold=0.5, mini_batch=mini_batch)

            batch_size, num_nodes = A.shape[0], A.shape[1]
            mask = torch.ones_like(A)

            indices = torch.arange(num_nodes, device=A.device)
            mask[:, indices, indices] = 0.0

            batch_size, n, n = logits_for_improvements.shape
            mask = ~torch.eye(n, dtype=torch.bool, device=logits_for_improvements.device).unsqueeze(0).expand(
                batch_size, -1, -1)
            # A가 1인 위치의 확률과 0인 위치의 (1-확률) 계산
            # 사칙연산만 사용하여 그래디언트가 잘 흐르도록 함
            selected_probs = A * logits_for_improvements + (1 - A) * (1 - logits_for_improvements)

            # 비대각 요소만 선택
            masked_probs = selected_probs.masked_select(mask).view(batch_size, -1)

            # 로그 도메인에서 합산 (수치적 안정성을 위해)
            log_probs = torch.log(masked_probs + 1e-8)  # 수치적 안정성을 위한 작은 값 추가
            summed_values = log_probs.sum(dim=1)

            """

            Original Version
            logits_for_improvements = torch.log(logits_for_improvements + 0.001)

            """

            A = A * mask
            I = torch.eye(h.shape[1])
            batch_identity = I.repeat(batch_size, 1, 1).to(device)
            A = A + batch_identity
            return A, summed_values

# class GLCN(nn.Module):
#     def __init__(self, feature_size,
#                  graph_embedding_size,
#                  feature_obs_size):
#         super(GLCN, self).__init__()
#         self.graph_embedding_size = graph_embedding_size
#         self.feature_obs_size = feature_obs_size
#         self.a_link = nn.Parameter(torch.empty(size=(self.feature_obs_size, 1)))
#         nn.init.xavier_uniform_(self.a_link.data, gain=1.414)
#         self.k_hop = int(os.environ.get("k_hop",2))
#         self.Ws = [nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
#         [glorot(W) for W in self.Ws]
#         self.a = [nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) if k == 0 else nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) for k in range(self.k_hop)]
#         [nn.init.xavier_uniform_(self.a[k].data, gain=1.414) for k in range(self.k_hop)]
#         self.Ws = nn.ParameterList(self.Ws)
#         self.a = nn.ParameterList(self.a)
#
#
#
#
#
#     def _link_prediction(self, h):
#         h = h.detach()
#         h = h[:, :self.feature_obs_size]
#         h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
#         h = h.squeeze(2)
#         A = gumbel_sigmoid(h, tau = float(os.environ.get("gumbel_tau",1.0)), hard = True, threshold = 0.5)
#         D = torch.diag(torch.diag(A))

#         A = A-D
#         I = torch.eye(A.size(0)).to(device)
#         A = A+I
#         return A
#
#     def _prepare_attentional_mechanism_input(self, Wq, Wv, k = None, mini_batch = False):
#         if mini_batch == False:
#             Wh1 = Wq
#             Wh1 = torch.matmul(Wh1, self.a[k][:self.graph_embedding_size, : ])
#             Wh2 = Wv
#             Wh2 = torch.matmul(Wh2, self.a[k][self.graph_embedding_size:, :])
#             e = Wh1 + Wh2.T
#         else:
#             Wh1 = Wq
#             Wh2 = Wv
#             Wh1 = torch.matmul(Wh1, self.a[k][:self.graph_embedding_size, :])
#             Wh2 = torch.matmul(Wh2, self.a[k][self.graph_embedding_size:, :])
#             e = Wh1 + torch.transpose(Wh2, 1, 2)
#         return F.leaky_relu(e, negative_slope=cfg.negativeslope)
#
#
#
#     def forward(self, A, X, dead_masking = False, mini_batch = False):
#         if mini_batch == False:
#             A = self._link_prediction(X)
#             H = X
#             for k in range(self.k_hop):
#                 Wh = H @ self.Ws[k]
#                 a = self._prepare_attentional_mechanism_input(Wh, Wh, k=k)
#                 zero_vec = -9e15 * torch.ones_like(A)
#                 a = torch.where(A > 0, A * a, zero_vec)
#                 a = F.softmax(a, dim=1)
#                 H = torch.matmul(a, Wh)
#             return H, A, X
#         else:
#             num_nodes = X.shape[1]
#             batch_size = X.shape[0]
#             Hs = torch.zeros([batch_size, num_nodes, self.graph_embedding_size]).to(device)
#             As = torch.stack([self._link_prediction(X[b]) for b in range(batch_size)])
#             H = X
#             A = As
#             for k in range(self.k_hop):
#                 Wh = H @ self.Ws[k]
#                 a = self._prepare_attentional_mechanism_input(Wh, Wh, k=k, mini_batch = mini_batch)
#                 zero_vec = -9e15 * torch.ones_like(A)
#                 a = torch.where(A > 0, A * a, zero_vec)
#                 a = F.softmax(a, dim=2)
#                 H = torch.matmul(a, Wh)
#                 if k + 1 == self.k_hop:
#                     Hs = H
#             return Hs, As, X, 1
