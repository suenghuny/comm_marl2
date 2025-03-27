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


def gumbel_sigmoid(logits: Tensor, tau: float = 1.0, hard: bool = True, threshold: float = 0.5,
                   mini_batch: bool = False,
                   start_factor = None,
                   step = None,
                   decaying_factor = None,
                   min_factor = None,
                    rollout = True
                   ) -> Tensor:
    if rollout == False:
        noise_scale_factor = np.max([start_factor-step*decaying_factor, min_factor])
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )
        gumbels = (logits + gumbels*noise_scale_factor) / tau  # ~Gumbel(logits, tau)
    else:
        gumbels = logits / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()
    logits_for_improvements = y_soft
    if hard:

        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        for i in range(y_soft.shape[0]):  # 배치 차원으로 루프
            indices = (y_soft[i] > threshold).nonzero(as_tuple=True)
            if indices[0].size(0) > 0:  # 인덱스가 존재하는 경우에만
                y_hard[i, indices[0], indices[1]] = 1.0
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


class GAT2(nn.Module):
    def __init__(self, feature_size, graph_embedding_size):
        super(GAT2, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.Ws = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
        self.Wq = nn.Parameter(torch.Tensor(feature_size, 1))
        self.Wk = nn.Parameter(torch.Tensor(feature_size, 1))
        glorot(self.Ws)
        glorot(self.Wq)
        glorot(self.Wk)
        self.a = nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(0.2)

    def _prepare_attentional_mechanism_input(self, Wq, Wv):
        Wh1 = Wq
        Wh2 = Wv
        e = Wh1 + torch.transpose(Wh2, 1, 2)
        return F.leaky_relu(e, negative_slope=cfg.negativeslope)

    def forward(self, A, X, dense=False):
        batch_size, num_nodes, feature_size = X.shape
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
        Wh = X @ self.Ws  # [batch_size, num_nodes, graph_embedding_size]
        Wq = X @ self.Wq  # [batch_size, num_nodes, graph_embedding_size]
        Wk = X @ self.Wk  # [batch_size, num_nodes, graph_embedding_size]
        e = self._prepare_attentional_mechanism_input(Wq,Wk)
        zero_vec = -9e15 * torch.ones_like(E)
        attention = torch.where(E > 0, E*e, zero_vec)
        attention = self.dropout(F.softmax(attention, dim=2))

        # 최종 노드 표현 계산
        H = F.elu(torch.matmul(attention, Wh))

        # 출력에 드롭아웃 적용
        H = self.dropout(H)
        return H


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) 구현

    GIN 레이어는 다음 수식을 기반으로 합니다:
    h_i^(k+1) = MLP^(k)((1 + ε^(k)) · h_i^(k) + ∑_{j∈N(i)} h_j^(k))
    """

    def __init__(self, feature_size, graph_embedding_size, num_layers=1, eps=0, train_eps=True):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.mlps = nn.ModuleList()

        # 학습 가능한 epsilon 파라미터 (논문의 ε)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = eps

        # 첫 번째 레이어
        self.mlps.append(self.create_mlp(feature_size, graph_embedding_size))

        # 중간 레이어들
        for i in range(num_layers - 2):
            self.mlps.append(self.create_mlp(graph_embedding_size, graph_embedding_size))


        # 마지막 레이어
        self.mlps.append(self.create_mlp(graph_embedding_size, graph_embedding_size))

    def create_mlp(self, in_dim, out_dim):
        """2층 MLP 생성"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, X, A, dense = False):
        """
        X: 노드 특성 행렬 [batch_size, num_nodes, input_dim] 또는 [num_nodes, input_dim]
        A: 인접 행렬 [batch_size, num_nodes, num_nodes] 또는 [num_nodes, num_nodes]
        """
        batch_size, num_nodes, feature_size = X.shape
        if dense == False:
            A = torch.stack([
                torch.sparse_coo_tensor(
                    torch.tensor(A[b], dtype=torch.long).to(device),
                    torch.ones(torch.tensor(A[b]).shape[1]).to(device),
                    (num_nodes, num_nodes)
                ).to_dense()
                for b in range(batch_size)
            ], dim=0)
        else:
            A = A

        # 배치 차원 처리
        if X.dim() == 2:
            X = X.unsqueeze(0)  # [num_nodes, input_dim] -> [1, num_nodes, input_dim]

        if A.dim() == 2:
            A = A.unsqueeze(0)  # [num_nodes, num_nodes] -> [1, num_nodes, num_nodes]

        # 자기 루프 추가 (필요한 경우)
        # self_loop = torch.eye(A.size(1), device=A.device).unsqueeze(0).expand_as(A)
        # A = A + self_loop  # 자기 루프가 이미 있는 경우 주석 처리

        h = X

        # GIN 레이어 적용
        for i in range(self.num_layers):
            # 이웃 노드 특성의 합 계산
            neighbor_sum = torch.bmm(A, h)

            # GIN 업데이트 규칙 적용: (1 + ε) * h_i + ∑_{j∈N(i)} h_j
            h = (1 + self.eps) * h + neighbor_sum

            # MLP와 배치 정규화 적용
            batch_size, num_nodes, feat_dim = h.size()
            h = h.view(-1, feat_dim)  # [batch_size * num_nodes, feat_dim]
            h = self.mlps[i](h)
            h = F.relu(h)
            h = h.view(batch_size, num_nodes, -1)  # 원래 형태로 복원

        return h

    def predict(self, X, A):
        """
        노드 분류를 위한 예측 함수
        """
        logits = self.forward(X, A)
        return F.log_softmax(logits, dim=2)

    def graph_pooling(self, h, batch_index, pooling_type="mean"):
        """
        그래프 분류를 위한 그래프 풀링 함수
        batch_index: 각 노드가 속한 그래프의 인덱스 [num_nodes]
        pooling_type: 'sum', 'mean', 'max' 중 하나
        """
        if h.dim() == 3:  # [batch_size, num_nodes, feat_dim]
            if pooling_type == "sum":
                return h.sum(dim=1)
            elif pooling_type == "mean":
                return h.mean(dim=1)
            elif pooling_type == "max":
                return h.max(dim=1)[0]
        else:
            raise NotImplementedError("배치 처리된 그래프 풀링은 구현되지 않았습니다.")


class GAT(nn.Module):
    def __init__(self, feature_size, graph_embedding_size):
        super(GAT, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.Ws = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
        glorot(self.Ws)
        self.a = nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(0.4)

    def _prepare_attentional_mechanism_input(self, Wq, Wv):
        Wh1 = Wq
        Wh2 = Wv
        Wh1 = torch.matmul(Wh1, self.a[:self.graph_embedding_size, :])
        Wh2 = torch.matmul(Wh2, self.a[self.graph_embedding_size:, :])
        e = Wh1 + torch.transpose(Wh2, 1, 2)
        return F.leaky_relu(e, negative_slope=cfg.negativeslope)

    def forward(self, A, X, dense=False):
        batch_size, num_nodes, feature_size = X.shape
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
        Wh = X @ self.Ws  # [batch_size, num_nodes, graph_embedding_size]
        e = self._prepare_attentional_mechanism_input(Wh, Wh)

        zero_vec = -9e15 * torch.ones_like(E)
        attention = torch.where(E > 0, E*e, zero_vec)
        attention = F.softmax(attention, dim=2)
        H = F.elu(torch.matmul(attention, Wh))
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

        self.start_factor = 1.0
        self.step = 0
        self.decaying_factor = 0.000003
        self.min_factor = 0.05


    # def cal_log_prob(self, h):
    #     h = h[:, :, :self.feature_obs_size].detach()
    #     h = torch.einsum("bijk,kl->bijl", torch.abs(h.unsqueeze(2) - h.unsqueeze(1)), self.a_link)
    #     h = h.squeeze(3)
    #     A, logits_for_improvements = gumbel_sigmoid(h, mini_batch=True)
    #     batch_size, n, n = logits_for_improvements.shape
    #     mask = ~torch.eye(n, dtype=torch.bool, device=logits_for_improvements.device).unsqueeze(0).expand(
    #         batch_size, -1, -1)
    #     selected_probs = A * logits_for_improvements + (1 -A) * (1 - logits_for_improvements)
    #     masked_probs = selected_probs.masked_select(mask).view(batch_size, -1)
    #     probs = masked_probs + 1e-8  # 수치적 안정성을 위한 작은 값 추가
    #     probs = torch.log(probs).sum(dim=1)
    #     A = A * mask
    #     I = torch.eye(h.shape[1])
    #     batch_identity = I.repeat(batch_size, 1, 1).to(device)
    #     A = A + batch_identity
    #     return A, probs

    def forward(self, h, rollout, check = False):
        if rollout==False:
            if check == True:
                self.step +=1
        h = h[:, :, :self.feature_obs_size].detach()
        h = torch.einsum("bijk,kl->bijl", torch.abs(h.unsqueeze(2) - h.unsqueeze(1)), self.a_link)
        h = h.squeeze(3)
        A, logits_for_improvements = gumbel_sigmoid(h, mini_batch=True, rollout=rollout,
                                                    start_factor=self.start_factor,
                                                    step = self.step,
                                                    decaying_factor= self.decaying_factor,
                                                    min_factor = self.min_factor
        )
        batch_size, n, n = logits_for_improvements.shape
        mask = ~torch.eye(n, dtype=torch.bool, device=logits_for_improvements.device).unsqueeze(0).expand(
            batch_size, -1, -1)
        selected_probs = A * logits_for_improvements + (1 -A) * (1 - logits_for_improvements)
        masked_probs = selected_probs.masked_select(mask).view(batch_size, -1)
        probs = masked_probs + 1e-8  # 수치적 안정성을 위한 작은 값 추가
        probs = torch.log(probs).sum(dim=1)

        batched_diag_matrices = torch.zeros_like(A)
        for i in range(batch_size):
            batched_diag_matrices[i,:,:]=torch.diag(torch.diag(A[i]))

        A = A - batched_diag_matrices
        I = torch.eye(h.shape[1])
        batch_identity = I.repeat(batch_size, 1, 1).to(device)
        A = A + batch_identity
        A = A.squeeze(0)
        return A, probs


        # h = h.detach()
        # h = h[:, :self.feature_obs_size]
        # h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
        # h = h.squeeze(2)
        # A, logits_for_improvements = gumbel_sigmoid(h)
        #
        # A = A.unsqueeze(0)
        # batch_size, n, n = A.shape
        # mask = ~torch.eye(n, dtype=torch.bool, device=logits_for_improvements.device).unsqueeze(0).expand(batch_size, -1, -1)
        # selected_probs = A * logits_for_improvements + (1 - A) * (1 - logits_for_improvements)
        # masked_probs = selected_probs.masked_select(mask).view(batch_size, -1)
        # probs = masked_probs + 1e-8  # 수치적 안정성을 위한 작은 값 추가
        # probs = torch.log(probs).sum()
        # A = A.squeeze(0)
        # A = A - torch.diag(torch.diag(A))
        # I = torch.eye(h.shape[1]).to(device)
        # A = A + I
        #