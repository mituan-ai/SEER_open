"""合成语义先验的观测变换。

一个变换把变量 j 的记录值替换为

    y_s       = a * sum_{l=0}^{lmax} h_l * x_{k, s-l} + b + sigma * eps_s
    x~_{j,s}  = y_{hold(s)},   hold(s) = s - ((s - phi) mod P)

k 为来源变量（可为 j 自身），h 为时滞核，a、b 为增益与偏置，P 为采样保持周期（P = 1 表示不保持），
phi 为保持相位。数值都在标准化尺度上。变换作用在逐步流上：输入窗口与下一步观测取自同一条变换后的序列，
因此二者一致；早于段起点的历史按段起点补齐。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Transforms:
    """N 个变换的参数；``kernel`` 为 (N, lmax + 1)。"""

    target: np.ndarray
    source: np.ndarray
    kernel: np.ndarray
    gain: np.ndarray
    bias: np.ndarray
    noise: np.ndarray
    hold: np.ndarray

    def __len__(self) -> int:
        return int(self.target.size)

    def take(self, index: np.ndarray) -> Transforms:
        return Transforms(*(getattr(self, name)[index] for name in self.__dataclass_fields__))


def apply_stream(values: np.ndarray, transforms: Transforms, row: int, *, phase: int = 0) -> np.ndarray:
    """对一段流 (T, d) 施加第 ``row`` 个变换，返回替换后的整段流。"""

    j, k = int(transforms.target[row]), int(transforms.source[row])
    kernel = transforms.kernel[row]
    steps = np.arange(len(values))
    lagged = values[np.maximum(steps[:, None] - np.arange(kernel.size), 0), k]
    rng = np.random.default_rng(row)
    y = (
        transforms.gain[row] * lagged @ kernel
        + transforms.bias[row]
        + transforms.noise[row] * rng.normal(size=len(values))
    )
    period = int(transforms.hold[row])
    out = values.copy()
    out[:, j] = y[np.maximum(steps - (steps - phase) % period, 0)]
    return out


def apply_windows(
    stream: Tensor,
    starts: Tensor,
    ends: Tensor,
    window: int,
    params: dict[str, Tensor],
    generator: torch.Generator | None = None,
) -> Tensor:
    """在 GPU 上为一批窗口生成变换后的记录。

    stream (T, d) 为标准化逐步流；starts、ends (B,) 为各窗口所在段的起点与窗口末端行号。
    ``params`` 含 target、source (B,)、kernel (B, lmax + 1)、gain、bias、noise、hold、phase (B,)。
    返回 (B, window + 1)：窗口 [t - L + 1, t] 与下一步 t + 1 上 x~_j 的值。
    """

    hold = params["hold"].long()
    span = int(hold.max())
    lmax = params["kernel"].shape[1] - 1
    block = ends[:, None] - window + 1 - (span - 1) + torch.arange(window + span, device=stream.device)
    lagged_rows = (
        (block[:, :, None] - torch.arange(lmax + 1, device=stream.device))
        .clamp(min=starts[:, None, None])
        .clamp(max=len(stream) - 1)
    )  # 段末窗口的 t + 1 不在段内，其值不参与损失
    source = stream[lagged_rows, params["source"].long()[:, None, None]]
    noise = torch.randn(block.shape, device=stream.device, generator=generator)
    y = params["gain"][:, None] * (source * params["kernel"][:, None, :]).sum(-1)
    y = y + params["bias"][:, None] + params["noise"][:, None] * noise
    positions = torch.arange(span - 1, span + window, device=stream.device)  # 块内 [t-L+1, t+1]
    absolute = block[:, span - 1 :]
    index = positions[None, :] - torch.remainder(absolute - params["phase"].long()[:, None], hold[:, None])
    return y.gather(1, index)
