"""SEER 语义读取层。

形状约定：B 为批次，L 为窗口长度，d 为变量数，F=4 为字段数，k 为嵌入维度，
H 为读取头数，Lr 为读取时滞数，m 为语义投影维度。

    e_i       = LayerNorm(W_e [e_{i,1}; ...; e_{i,F}])                         (B, d, m)
    s^h_{ijl} = q^h_i . k^h_{jl} + e_i^T U^h_l e_j
    q^h_i     = W^h_q e_i,   k^h_{jl} = W^h_k e_j + p^h_l
    U^h_l     = A^h^T diag(u^h_l) C^h                                          低秩，(m, m)
    a^h_{ijl} = softmax_{j,l}(s^h_{ijl} / sqrt(m))                             (B, H, d, d, Lr)
    r^h_{i,t} = sum_{j,l} a^h_{ijl} x_{j,t-l}                                  (B, L, d, H)

查询与键只由语义和时滞位置生成，不读取数值，所以读取核完全由语义决定：
同一套语义下，读取层是按变量的因果卷积，可以接在任何主干之前。
语义决定“这一列与谁、按什么时滞关联”，同时刻的跨变量作用（l = 0）、延迟与来源变化都由同一个核表达。
窗口开头不足 Lr - 1 步的历史用第一步补齐。

观测头（``ObservationHead``）读取主干状态与卡片的原始字段嵌入，给出下一步记录的条件分布，
可在链式顺序下读取同时刻已揭示的记录（``visibility``）；目标通路从不读取 t + 1。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SemanticKernel(nn.Module):
    """语义打分 s^h_{ijl} = (q^h_i . k^h_{jl} + e_i^T U^h_l e_j) / sqrt(m)，形状 (B, H, n_i, d_j, Lr)。"""

    def __init__(self, *, heads: int, lags: int, semantic_dim: int) -> None:
        super().__init__()
        self.heads, self.lags = int(heads), int(lags)
        m = int(semantic_dim)
        self.query = nn.Linear(m, heads * m, bias=False)
        self.key = nn.Linear(m, heads * m, bias=False)
        self.lag_key = nn.Parameter(torch.randn(heads, lags, m) * m**-0.5)
        self.left = nn.Linear(m, heads * m, bias=False)
        self.right = nn.Linear(m, heads * m, bias=False)
        self.lag_bilinear = nn.Parameter(torch.randn(heads, lags, m) * m**-0.5)

    def forward(self, rows: Tensor, e: Tensor) -> Tensor:
        """rows (B, n, m) 为查询变量的语义，e (B, d, m) 为全部变量的语义。"""

        m = e.shape[-1]

        def split(linear: nn.Linear, x: Tensor) -> Tensor:
            return linear(x).view(*x.shape[:2], self.heads, m).transpose(1, 2)  # (B, H, n, m)

        q, left = split(self.query, rows), split(self.left, rows)
        k, right = split(self.key, e), split(self.right, e)
        logits = torch.einsum("bhim,bhjm->bhij", q, k).unsqueeze(-1)  # 跨变量项
        logits = logits + torch.einsum("bhim,hlm->bhil", q, self.lag_key).unsqueeze(-2)  # 查询-时滞项
        logits = logits + torch.einsum("bhim,hlm,bhjm->bhijl", left, self.lag_bilinear, right)
        return logits / math.sqrt(m)


class SemanticReader(nn.Module):
    def __init__(self, *, fields: int, embed_dim: int, lags: int, heads: int, semantic_dim: int) -> None:
        super().__init__()
        self.lags, self.heads, self.semantic_dim = int(lags), int(heads), int(semantic_dim)
        self.combine = nn.Linear(fields * embed_dim, semantic_dim)
        self.norm = nn.LayerNorm(semantic_dim)
        self.score = SemanticKernel(heads=heads, lags=lags, semantic_dim=semantic_dim)

    def semantics(self, fields: Tensor) -> Tensor:
        """(B, d, F, k) 字段嵌入 -> (B, d, m) 变量语义。"""

        return self.norm(self.combine(fields.flatten(-2)))

    def kernel(self, e: Tensor) -> Tensor:
        """读取核 a，形状 (B, H, d_i, d_j, Lr)。"""

        logits = self.score(e, e)
        return F.softmax(logits.flatten(-2), dim=-1).view_as(logits)

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        """x (B, L, d) 与读取核 a (B, H, d, d, Lr) -> 读取序列 (B, L, d * H)：按变量的因果卷积。"""

        padded = torch.cat([x[:, :1].expand(-1, self.lags - 1, -1), x], dim=1)
        lagged = padded.unfold(1, self.lags, 1).flip(-1)  # (B, L, d, Lr)，最后一维为时滞 0, 1, ...
        r = torch.einsum("bhijl,btjl->btih", a, lagged)
        return r.flatten(-2)


class ObservationHead(nn.Module):
    """观测头：x_{i,t+1} 在 F_t 与同时刻已揭示记录下的 Student-t 条件分布（标准化尺度）。

    读取权重 w_{ijl} 作用于各（变量，时滞）的一步增量（l 相对 t + 1）：

        mu_i      = x_{i,t} + sum_{j,l} w_{ijl} dx_{j,t+1-l} + beta_i,
        sigma_i^2 = (softplus(s_i) + floor_i)^2 + sum_{j 不可见} w_{ij0}^2 v_j^2,
        w_{ijl}   = n_{ijl} v_i / v_j + sum_f pi^f_{ij} g^f_{ij} r^f_{il}.

    - 自然读取 n：按（观测变量，变量，时滞）的自由参数，由训练数据估计原语义下记录间的增量关系，
      与主干对各变量的输入权重同类，不经语义生成。以一步增量尺度预条件：w = n v_i / v_j，n 是“来源的一步
      典型增量带来多少目标的一步典型增量”，量级为 1（积分型记录的一步增量极小，原尺度的权重可达数百）。
      语义生成的读取（压缩语义上的双线性分数、各来源共享一个非负时滞形状）表达不了稀疏、带符号、
      各来源时滞不同的自然依赖。
    - 文本引用 pi^f_{ij} g^f_{ij} r^f_{il}：pi^f_{ij} = softmax_{j != i}(tau_f cos(E_{i,f}, E_{j,来源}))，
      f 取陈述关系的两个字段——计算（记录由什么算出）与依赖（什么驱动它）——在编码器原始嵌入空间中
      提到了哪个变量（其来源字段：它是什么）；字段对由卡片格式决定，余弦不含按变量的参数，每个字段只有
      一个温度。卡片改写时 pi 随文本移到新来源上。带符号的成对增益 g 与时滞核 r 由原始字段的投影
      z = LayerNorm(W_z E) 给出，只由观测似然训练。

    dx_{j,t+1-l} = x_{j,t+1-l} - x_{j,t-l}；不可见的同时刻增量（被预测变量自身与链式顺序中尚未揭示者）
    按随机游走边缘化：均值取 0，方差 v_j^2 为训练段一步增量的均方。beta、s 由 h_t 给出；w = 0 即持续性。
    标准差下界 floor 由训练段一步增量确定后冻结（阶梯式记录的一步增量多为零）。
    """

    def __init__(
        self,
        hidden: int,
        observed: tuple[int, ...],
        floor: np.ndarray,
        step: np.ndarray,
        *,
        variables: int,
        fields: int,
        embed_dim: int,
        reference: tuple[tuple[int, ...], int],
        lags: int,
        semantic_dim: int,
    ) -> None:
        super().__init__()
        self.register_buffer("observed", torch.as_tensor(observed, dtype=torch.long))
        self.register_buffer("floor", torch.as_tensor(floor, dtype=torch.float32))
        self.register_buffer("step", torch.as_tensor(step, dtype=torch.float32))  # (d,)
        self.lags = int(lags)
        self.natural = nn.Parameter(torch.zeros(len(observed), variables, lags))  # n
        self.queries, self.key = reference  # 陈述关系的字段（计算、依赖）与来源字段
        Q, m = len(self.queries), semantic_dim
        self.temperature = nn.Parameter(torch.full((Q,), math.log(20.0)))  # log tau，各字段一个
        self.project = nn.Sequential(nn.Linear(fields * embed_dim, m), nn.LayerNorm(m))
        self.gain = nn.ModuleList(SemanticKernel(heads=1, lags=1, semantic_dim=m) for _ in range(Q))
        self.delay = nn.ModuleList(nn.Linear(m, lags) for _ in range(Q))
        self.linear = nn.Linear(hidden, 2 * len(observed))
        self.dof = nn.Parameter(torch.full((len(observed),), math.log(8.0)))  # log(nu - 2)

    def references(self, fields: Tensor, q: int = 0) -> Tensor:
        """第 q 个关系字段的文本引用 pi (B, J, d)：原始字段嵌入 (B, d, F, k) 上的余弦注意力，排除自身。"""

        unit = F.normalize(fields, dim=-1)
        cos = torch.einsum("bik,bjk->bij", unit[:, self.observed, self.queries[q]], unit[:, :, self.key])
        logits = torch.exp(self.temperature[q]) * cos
        own = F.one_hot(self.observed, fields.shape[1]).bool()
        return F.softmax(logits.masked_fill(own, -torch.inf), dim=-1)

    def weights(self, fields: Tensor) -> Tensor:
        """读取权重 w = n v_i / v_j + sum_f pi^f g^f r^f，形状 (B, J, d, Lr)：只依赖卡片。"""

        z = self.project(fields.flatten(-2))
        rows = z[:, self.observed]
        v = self.step
        ratio = torch.where(v > 0, v[self.observed, None] / v.clamp_min(1e-12)[None, :], 0.0)  # (J, d)
        w = (self.natural * ratio[..., None]).expand(len(fields), -1, -1, -1)
        for q, (gain, delay) in enumerate(zip(self.gain, self.delay, strict=True)):
            reference = gain(rows, z)[:, 0, :, :, 0] * self.references(fields, q)
            w = w + reference[..., None] * F.softmax(delay(rows), dim=-1)[:, :, None, :]
        return w

    def forward(
        self,
        h: Tensor,
        w: Tensor,
        x: Tensor,
        following: Tensor | None,
        visible: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """w (B, J, d, Lr) 为读取权重（``weights``）；visible (B 或 1, J, d)：预测第 j 个观测变量时
        哪些变量的 x_{t+1} 可见；None 时都不可见。"""

        lags, last = self.lags, x[:, -1]
        if following is None or visible is None:
            following = last
            visible = torch.zeros(1, len(self.observed), x.shape[-1], dtype=torch.bool, device=x.device)
        now = (following - last)[:, None] * visible  # (B, J, d)
        steps = torch.diff(torch.cat([x[:, :1].expand(-1, lags, -1), x], dim=1), dim=1)
        past = steps[:, steps.shape[1] - (lags - 1) :].flip(1)  # (B, Lr - 1, d)：dx_t, dx_{t-1}, ...
        read = torch.einsum("bij,bij->bi", w[..., 0], now) + torch.einsum("bijl,blj->bi", w[..., 1:], past)
        hidden = ~visible[..., self.observed]  # (B, J, J)；自身对应的权重乘以零增量，同样边缘化
        v = self.step[self.observed]
        missing = torch.sum(w[..., 0][..., self.observed] ** 2 * hidden * v**2, dim=-1)
        beta, scale = self.linear(h).chunk(2, dim=-1)
        mean = last[:, self.observed] + read + beta
        return mean, torch.sqrt((F.softplus(scale) + self.floor) ** 2 + missing)

    def nll(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """观测似然：各观测变量一个自由度 nu = 2 + exp(.)（训练学习）的 Student-t。过程记录的一步残差
        重尾（排放、补料、阶跃），高斯似然的尺度被尖峰抬高后，均值在大量平常步上几乎得不到梯度。"""

        return student_nll(x, mean, scale, 2 + torch.exp(self.dof))


def visibility(observed: Tensor, variables: int, tested: Tensor) -> Tensor:
    """部署时的同时刻可见性 (J, d)：被检验集 U（bool (J,)）按变量序链式揭示。

    i 在 U 中：可见 = U 之外的全部变量与 U 中序在 i 之前者；i 不在 U 中：除自身外全部可见（留一）。
    同一 U 下各被检验因子之积是 x_{U,t+1} 在 F_t 与 x_{-U,t+1} 下的条件联合密度。
    """

    J = len(observed)
    order = torch.arange(J, device=observed.device)
    hidden = (order[None, :] >= order[:, None]) & tested[None, :] & tested[:, None]  # (J_i, J_j)
    hidden |= torch.eye(J, dtype=torch.bool, device=observed.device)
    visible = torch.ones(J, variables, dtype=torch.bool, device=observed.device)
    visible[:, observed] = ~hidden
    return visible


def random_visibility(
    observed: Tensor, variables: int, include: Tensor, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """训练时的可见性 (B, J, d) 与被检验集 U (B, J)：从部署时同一族中抽取，U 的大小在 1..J 上均匀、
    成员与顺序随机，并含 ``include``（bool (B, J)，部署时 U 必含改动的变量）。"""

    batch, J = include.shape
    device = observed.device
    rank = torch.rand(batch, J, device=device, generator=generator).argsort(dim=1).argsort(dim=1)
    size = torch.randint(1, J + 1, (batch, 1), device=device, generator=generator)
    tested = (rank < size) | include
    hidden = (rank[:, None, :] >= rank[:, :, None]) & tested[:, None, :] & tested[:, :, None]
    hidden |= torch.eye(J, dtype=torch.bool, device=device)
    visible = torch.ones(batch, J, variables, dtype=torch.bool, device=device)
    visible[:, :, observed] = ~hidden
    return visible, tested


def student_nll(x: Tensor, mean: Tensor, scale: Tensor, nu: Tensor) -> Tensor:
    """Student-t 负对数密度（位置 mean、尺度 scale、自由度 nu）；nu -> inf 即高斯。"""

    z = (x - mean) / scale
    return (
        torch.lgamma(nu / 2)
        - torch.lgamma((nu + 1) / 2)
        + 0.5 * torch.log(nu * math.pi)
        + torch.log(scale)
        + (nu + 1) / 2 * torch.log1p(z**2 / nu)
    )
