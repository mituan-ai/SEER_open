from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seer.methods.seer import visibility
from seer.models.forecaster import build_forecaster

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = yaml.safe_load((ROOT / "configs" / "backbones.yaml").read_text(encoding="utf-8"))
NAMES = ("a", "b", "c", "d")


FIELDS = torch.randn(len(NAMES), 4, 32, generator=torch.Generator().manual_seed(0)).numpy()
SEER = {"name": "seer", "reader_heads": 2, "semantic_dim": 8}


def _model(method: str, backbone: str, window: int = 16):
    config = SEER if method == "seer" else {"name": "base"}
    return build_forecaster(
        method=config,
        backbone=BACKBONES[backbone],
        feature_names=NAMES,
        window_size=window,
        output_dim=3,
        fields=FIELDS,
        observed=(0, 2),
        scale_floor=np.full(2, 1e-3),
        step=np.ones(len(NAMES)),
        reference=((1, 3), 0),
        reader_lags=4,
    )


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
@pytest.mark.parametrize("method", ["base", "seer"])
def test_every_backbone_reads_every_method_output(backbone: str, method: str) -> None:
    prediction, observed = _model(method, backbone)(torch.randn(2, 16, len(NAMES)))
    assert prediction.shape == (2, 3)
    assert (observed is None) == (method == "base")
    if observed is not None:
        assert observed[0].shape == observed[1].shape == (2, 2) and (observed[1] > 0).all()


def test_reader_is_causal_and_semantics_change_the_kernel() -> None:
    torch.manual_seed(0)
    model = _model("seer", "gru").eval()
    x = torch.randn(1, 16, len(NAMES), requires_grad=True)
    read = model.reader(x, model.reader.kernel(model.reader.semantics(model.fields[None])))
    grad = torch.autograd.grad(read[:, 7].sum(), x)[0]
    assert grad[:, 8:].abs().max() == 0 and grad[:, 4:8].abs().max() > 0  # 只读 t-3..t
    kernel = model.reader.kernel(model.reader.semantics(model.fields[None]))
    assert torch.allclose(kernel.sum((-1, -2)), torch.ones(1, 2, 4))
    other = model.fields.clone()
    other[1, 2] = torch.randn(32)
    fields = torch.stack([model.fields, other])
    kernel = model.reader.kernel(model.reader.semantics(fields))
    assert not torch.allclose(kernel[0], kernel[1])
    assert not torch.allclose(
        model(x.detach().expand(2, -1, -1), fields)[0][0], model(x.detach().expand(2, -1, -1), fields)[0][1]
    )


def test_observation_head_reads_only_revealed_records() -> None:
    """观测头在链式顺序下只读取已揭示的同时刻记录；目标预测不读取 t + 1。"""

    torch.manual_seed(0)
    model = _model("seer", "gru").eval()
    observed = model.observation.observed
    x = torch.randn(1, 16, len(NAMES))
    following = torch.randn(1, len(NAMES), requires_grad=True)
    visible = visibility(observed, len(NAMES), torch.tensor([True, True]))[None]  # U = 全部观测变量 (0, 2)
    prediction, (mean, _) = model(x, None, following, visible)
    assert torch.autograd.grad(prediction.sum(), following, allow_unused=True)[0] is None
    first = torch.autograd.grad(mean[:, 0].sum(), following, retain_graph=True)[0][0]
    second = torch.autograd.grad(mean[:, 1].sum(), following)[0][0]
    assert first[[0, 2]].abs().max() == 0 and first[[1, 3]].abs().min() > 0  # 0 号先揭示：看不到 0 与 2
    assert second[2] == 0 and second[[0, 1, 3]].abs().min() > 0  # 2 号在 0 号之后：可见 0


def _outputs(model, x, fields=None, following=None, visible=None):
    prediction, (mean, scale) = model(x, fields, following, visible)
    return prediction, mean, scale


def _assert_same(first, second) -> float:
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    return max(float((a - b).abs().max()) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("batch", [1, 5, 33])
def test_default_semantics_are_computed_once_and_match_explicit_fields(batch: int) -> None:
    """推理时默认语义的读取核与观测权重算一次复用，与逐样本显式传入同一语义的结果一致。"""

    torch.manual_seed(0)
    model = _model("seer", "gru").eval()
    x, following = torch.randn(batch, 16, len(NAMES)), torch.randn(batch, len(NAMES))
    visible = visibility(model.observation.observed, len(NAMES), torch.tensor([True, False]))[None]
    explicit = model.fields.expand(batch, *model.fields.shape)
    with torch.no_grad():
        cached = _outputs(model, x, None, following, visible)
        state = model._frozen[2]
        assert _outputs(model, x[:1])[0].shape == (1, 3) and model._frozen[2] is state  # 命中，未重建
        direct = _outputs(model, x, explicit, following, visible)
        for a, b in zip(state, model.semantic_state(explicit), strict=True):
            torch.testing.assert_close(a.expand_as(b), b, rtol=0, atol=1e-6)
    _assert_same(cached, direct)


def test_cache_follows_parameter_and_semantic_updates() -> None:
    """优化器更新、load_state_dict、改写默认语义与 deepcopy 后，推理结果等于重新计算。"""

    import copy

    torch.manual_seed(0)
    model = _model("seer", "gru").eval()
    x = torch.randn(4, 16, len(NAMES))
    explicit = lambda: model.fields.expand(4, *model.fields.shape)  # noqa: E731

    def check(change) -> None:
        with torch.no_grad():
            before = _outputs(model, x)
        change()
        with torch.no_grad():
            after, direct = _outputs(model, x), _outputs(model, x, explicit())
        assert not torch.allclose(before[1], direct[1])  # 修改确实改变了结果
        _assert_same(after, direct)

    for prefix in ("reader.", "observation."):

        def step(prefix=prefix) -> None:
            chosen = [p for name, p in model.named_parameters() if name.startswith(prefix)]
            optimizer = torch.optim.SGD(chosen, lr=0.5)
            model.train()
            loss = sum(t.pow(2).sum() for t in model(x, None, torch.randn(4, len(NAMES)))[1])
            loss = loss + model(x)[0].pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()

        check(step)
    other = _model("seer", "gru").state_dict()
    check(lambda: model.load_state_dict(other))
    check(lambda: model.fields.copy_(torch.randn_like(model.fields)))
    with torch.no_grad():
        _outputs(model, x)
        twin = copy.deepcopy(model)
        twin.fields.add_(1.0)
        _assert_same(_outputs(twin, x), _outputs(twin, x, twin.fields.expand(4, -1, -1, -1)))
        _assert_same(_outputs(model, x), _outputs(model, x, explicit()))


def test_replaced_semantics_bypass_the_cache() -> None:
    """批内不同语义（在线候选、合成先验）、训练模式与求梯度时不使用缓存。"""

    torch.manual_seed(0)
    model = _model("seer", "gru").eval()
    x = torch.randn(2, 16, len(NAMES))
    other = model.fields.clone()
    other[2, 1] = torch.randn(32)  # 合成先验式的替换：一个观测变量的一个字段
    fields = torch.stack([model.fields, other])
    with torch.no_grad():
        _outputs(model, x)  # 建立默认语义的缓存
        mixed = _outputs(model, x, fields)
        alone = [_outputs(model, x[i : i + 1], fields[i : i + 1]) for i in range(2)]
    assert not torch.allclose(mixed[1][0], mixed[1][1])
    for i in range(2):
        _assert_same([t[i : i + 1] for t in mixed], alone[i])
    model._frozen = None
    model.train()
    with torch.no_grad():
        model(x)
    model.eval()
    model(x)[0].sum().backward()  # eval 但求梯度：不缓存，梯度到达读取层
    assert model._frozen is None
    assert model.reader.combine.weight.grad.abs().max() > 0


@pytest.mark.parametrize("window", [30, 60, 120])
def test_patchtst_reads_the_latest_steps_and_keeps_channels_independent(window: int) -> None:
    torch.manual_seed(0)
    backbone = build_forecaster(
        method={"name": "base"},
        backbone=BACKBONES["patchtst"],
        feature_names=NAMES,
        window_size=window,
        output_dim=1,
    ).backbone.eval()
    x = torch.randn(2, window, len(NAMES), requires_grad=True)
    out = backbone(x)
    assert torch.autograd.grad(out.sum(), x, retain_graph=True)[0][:, -1].abs().min() > 0
    blocks = out.reshape(2, len(NAMES), -1)
    grad_other = torch.autograd.grad(blocks[:, 0].sum(), x)[0][..., 1:]
    assert grad_other.abs().max() == 0  # 变量 0 独立编码，不混入其他变量


def test_patchtst_patch_positions_matter() -> None:
    torch.manual_seed(0)
    backbone = build_forecaster(
        method={"name": "base"},
        backbone=BACKBONES["patchtst"],
        feature_names=NAMES,
        window_size=32,
        output_dim=1,
    ).backbone.eval()
    x = torch.randn(1, 32, len(NAMES))
    swapped = torch.cat((x[:, 16:], x[:, :16]), dim=1)  # 相同分块采用不同顺序
    assert not torch.allclose(backbone(x).sum(dim=-1), backbone(swapped).sum(dim=-1))


def test_itransformer_tokens_have_no_variable_embedding() -> None:
    """官方 iTransformer 没有位置或变量编码：交换输入变量只交换对应的输出 token。"""

    torch.manual_seed(0)
    backbone = build_forecaster(
        method={"name": "base"},
        backbone=BACKBONES["itransformer"],
        feature_names=NAMES,
        window_size=16,
        output_dim=1,
    ).backbone.eval()
    x = torch.randn(2, 16, len(NAMES))
    order = torch.tensor([2, 0, 3, 1])
    tokens = backbone(x).reshape(2, len(NAMES), -1)
    permuted = backbone(x[..., order]).reshape(2, len(NAMES), -1)
    assert torch.allclose(permuted, tokens[:, order], atol=1e-5)


def test_text_reference_follows_the_variable_the_card_names() -> None:
    """文本引用 pi 由原始字段嵌入的余弦给出：卡片复述哪个变量的描述，pi 就落在哪个变量上，改写即移动。"""

    head = _model("seer", "gru").observation
    fields = torch.randn(1, len(NAMES), 4, 32, generator=torch.Generator().manual_seed(1))
    for source in (1, 3):
        named = fields.clone()
        named[0, 0, 1] = fields[0, source, 0]  # 观测变量 0 的计算字段复述变量 source 的来源字段
        pi = head.references(named)[0, 0]
        assert int(pi.argmax()) == source and pi[0] == 0  # 排除自身
