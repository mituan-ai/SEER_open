"""预测器 = [语义读取层] + 主干 + 目标头 [+ 观测头]。

Base：主干直接读取标准化窗口，只有目标头。
SEER：语义读取层把窗口与变量语义读成 (B, L, d * H) 的序列交给主干；目标头 g 与观测头 o 共用变量语义 e(S)
与最后的表示 h_t，观测头另有自己的生成读取核（可读取同时刻已揭示的记录）。
默认语义（原语义卡的字段嵌入）存为缓冲区；``forward`` 的 ``fields`` 参数可逐样本替换语义，不改参数。
读取核与观测头读取权重只由语义决定（``semantic_state``）：默认语义下的推理复用一次算好的结果，
读取层即固定的因果卷积。

Base 与 SEER 的主干输入维度不同（d 与 d * H），初始化不逐参数可比；两者的批次顺序由同一种子的独立生成器决定。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from seer.methods.seer import ObservationHead, SemanticReader
from seer.models.backbones import build_backbone

METHODS = ("base", "seer")


class Forecaster(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: nn.Module,
        reader: SemanticReader | None = None,
        observation: ObservationHead | None = None,
        fields: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.reader = reader
        self.backbone = backbone
        self.head = head
        self.observation = observation
        if fields is not None:
            self.register_buffer("fields", torch.as_tensor(np.asarray(fields), dtype=torch.float32))
        self._frozen: tuple | None = None  # (键, 持有的张量, 默认语义下的 (a, w))

    @property
    def semantic(self) -> bool:
        return self.reader is not None

    def forward(
        self,
        x: Tensor,
        fields: Tensor | None = None,
        following: Tensor | None = None,
        visible: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """返回目标预测与观测头的 (mu, sigma)；Base 的后者为 None。

        ``following`` (B, d) 为下一步记录，``visible`` (B 或 1, J, d) 为观测头可读取的同时刻记录；
        目标预测不依赖二者。观测头读取 h_t 与卡片的原始字段嵌入，不向主干与语义投影反传：
        二者只服务目标。
        """

        if self.reader is None:
            return self.head(self.backbone(x)), None
        batch = x.shape[0]
        if fields is None and not (self.training or torch.is_grad_enabled()):
            a, w = (t.expand(batch, *t.shape[1:]) for t in self._default_state())
        else:
            if fields is None:
                fields = self.fields.expand(batch, *self.fields.shape)
            a, w = self.semantic_state(fields)
        h = self.backbone(self.reader(x, a))
        return self.head(h), self.observation(h.detach(), w, x, following, visible)

    def semantic_state(self, fields: Tensor) -> tuple[Tensor, Tensor]:
        """字段嵌入 (B, d, F, k) -> 读取核 a (B, H, d, d, Lr) 与观测头读取权重 w (B, J, d, Lr)。"""

        return self.reader.kernel(self.reader.semantics(fields)), self.observation.weights(fields)

    def _default_state(self) -> tuple[Tensor, Tensor]:
        """默认语义下的 (a, w)，批维为 1；只在推理（非训练、不求梯度）时使用。

        键为参与计算的每个张量（读取层与观测头的参数和缓冲区、默认语义）的对象、地址与原位修改版本号：
        优化器更新、load_state_dict、设备与类型转换、替换缓冲区都会使其失效。缓存持有这些张量，
        对象标识不会被复用。
        """

        tensors = (
            *self.reader.parameters(),
            *self.observation.parameters(),
            *self.observation.buffers(),
            self.fields,
        )
        key = tuple((id(t), t.data_ptr(), t._version) for t in tensors)
        if self._frozen is None or self._frozen[0] != key:
            self._frozen = (key, tensors, self.semantic_state(self.fields[None]))
        return self._frozen[2]


def build_forecaster(
    *,
    method: dict[str, Any],
    backbone: dict[str, Any],
    feature_names: tuple[str, ...],
    window_size: int,
    output_dim: int,
    fields: np.ndarray | None = None,
    observed: tuple[int, ...] = (),
    scale_floor: np.ndarray | None = None,
    step: np.ndarray | None = None,
    reference: tuple[tuple[int, ...], int] | None = None,  # SEER：陈述关系的字段与来源字段的序号
    reader_lags: int = 1,
) -> Forecaster:
    name = method["name"]
    if name not in METHODS:
        raise ValueError(f"Unsupported method {name!r}")
    heads = int(method.get("reader_heads", 1))
    width = len(feature_names) * (heads if name == "seer" else 1)
    backbone_config = dict(backbone)
    kind = str(backbone_config.pop("type"))
    backbone_module = build_backbone(kind, input_dim=width, window_size=window_size, config=backbone_config)
    head = nn.Linear(int(backbone_module.output_dim), output_dim)
    if name == "base":
        return Forecaster(backbone=backbone_module, head=head)
    reader = SemanticReader(
        fields=int(fields.shape[1]),
        embed_dim=int(fields.shape[2]),
        lags=reader_lags,
        heads=heads,
        semantic_dim=int(method["semantic_dim"]),
    )
    observation = ObservationHead(
        int(backbone_module.output_dim),
        observed,
        scale_floor,
        step,
        variables=len(feature_names),
        fields=int(np.asarray(fields).shape[1]),
        embed_dim=int(np.asarray(fields).shape[2]),
        reference=reference,
        lags=reader_lags,
        semantic_dim=int(method["semantic_dim"]),
    )
    return Forecaster(
        backbone=backbone_module, head=head, reader=reader, observation=observation, fields=fields
    )
