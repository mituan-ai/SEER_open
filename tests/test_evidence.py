from __future__ import annotations

import math

import numpy as np

from seer.methods.evidence import Monitor, Prequential, Profile, belief, event_mass, features, risk


def _stream(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(n, 2))
    scale = rng.uniform(0.5, 2.0, size=(n, 2))
    current = rng.normal(size=(n, 2))
    following = 0.8 * mean + 0.3 * current + 0.1 + scale * 1.3 * rng.normal(size=(n, 2))
    return mean, scale, following, current


def test_prequential_product_equals_the_closed_form_marginal_likelihood() -> None:
    mean, scale, following, current = _stream(40)
    model = Prequential((2,), alpha=2.0, beta=1.0, floor=0.0)
    total = np.zeros(2)
    for t in range(40):
        phi, y = features(mean[t], scale[t], following[t], current[t])
        log_density, _ = model.update(phi, y, scale[t])
        total += log_density
    phi, y = features(mean, scale, following, current)
    for j in range(2):
        a0, b0, lam0, m0 = 2.0, 1.0, np.eye(3), np.array([1.0, 0.0, 0.0])
        lam = lam0 + phi[:, j].T @ phi[:, j]
        m = np.linalg.solve(lam, lam0 @ m0 + phi[:, j].T @ y[:, j])
        an = a0 + 40 / 2
        bn = b0 + 0.5 * (y[:, j] @ y[:, j] + m0 @ lam0 @ m0 - m @ lam @ m)
        closed = (
            math.lgamma(an)
            - math.lgamma(a0)
            + a0 * math.log(b0)
            - an * math.log(bn)
            + 0.5 * (np.linalg.slogdet(lam0)[1] - np.linalg.slogdet(lam)[1])
            - 20 * math.log(2 * math.pi)
            - np.log(scale[:, j]).sum()
        )
        assert math.isclose(total[j], closed, rel_tol=1e-9)


def test_profile_is_the_unpenalized_maximum_likelihood_in_the_original_space() -> None:
    mean, scale, following, current = _stream(30, seed=1)
    profile = Profile((2,), 0.0)
    for t in range(30):
        phi, y = features(mean[t], scale[t], following[t], current[t])
        profile.update(phi, y, scale[t])
    phi, y = features(mean, scale, following, current)
    expected = np.zeros(2)
    for j in range(2):
        theta = np.linalg.lstsq(phi[:, j], y[:, j], rcond=None)[0]
        c2 = np.mean((y[:, j] - phi[:, j] @ theta) ** 2)
        center = mean[:, j] * theta[0] + current[:, j] * theta[1] + theta[2]
        variance = c2 * scale[:, j] ** 2
        expected[j] = np.sum(
            -0.5 * np.log(2 * math.pi * variance) - (following[:, j] - center) ** 2 / (2 * variance)
        )
    np.testing.assert_allclose(profile.log_likelihood(), expected, rtol=1e-9)


def test_a_candidate_inside_the_original_family_never_gains_evidence() -> None:
    """性质 2：候选与原语义输出相同时，混合似然不超过轮廓最大似然，逐路径 E <= 1。"""

    mean, scale, following, current = _stream(60, seed=2)
    model, profile = Prequential((1, 2), 2.0, 1.0, 0.0), Profile((2,), 0.0)
    log_m, tested = np.zeros((1, 2)), np.ones(2, dtype=bool)
    for t in range(60):
        phi, y = features(mean[t], scale[t], following[t], current[t])
        log_m += model.update(phi[None], y[None], scale[t][None])[0]
        profile.update(phi, y, scale[t])
        b = belief(log_m, profile.log_likelihood(), tested, np.ones(1), 0.05)
        assert b.log_e <= 1e-9 and b.w <= 0.05 + 1e-12


def test_belief_scores_every_candidate_on_the_common_tested_set() -> None:
    log_m = np.array([[1.0, 9.0], [3.0, 2.0]])
    tested = np.array([True, False])
    b = belief(log_m, np.array([0.5, 8.0]), tested, np.array([0.5, 0.5]), 0.05)
    assert math.isclose(b.w / (1 - b.w), 0.05 / 0.95 * math.exp(b.log_e), rel_tol=1e-12)
    assert math.isclose(math.exp(b.log_e), 0.5 * math.exp(0.5) + 0.5 * math.exp(2.5), rel_tol=1e-12)
    assert np.isclose(b.weights().sum(), 1.0) and b.shares[1] > b.shares[0]
    assert math.isclose(risk(0.05, 0.5), 0.05 / 0.95)
    budgets = [event_mass(0.05, 0.5, e) for e in range(1, 2000)]
    assert all(math.isclose(risk(pi, 0.5), eps) for eps, pi in budgets)
    assert sum(eps for eps, _ in budgets) < 0.05


def test_monitor_takes_the_persistent_misfit_of_its_warmup_as_normal() -> None:
    """预热期已有的失配是当前语义的常态（不越界）；预热后才出现的失配越界。残差以 sigma 计量，
    不受重尾尖峰的方差估计影响。"""

    rng = np.random.default_rng(0)

    def fires(r: np.ndarray) -> bool:
        watch = Monitor(np.ones(1), 20, 2.0, 1.0, 0.1, cap=9.0, drift=1.0, threshold=30.0)
        mean, scale = np.zeros(1), np.ones(1)
        hit = False
        for t, e in enumerate(r):
            if t % 50 == 25:
                e = e + 100.0  # 偶发尖峰
            hit |= watch.update(*features(mean, scale, np.array([e]), np.zeros(1)), scale)
        return hit

    calm = rng.normal(size=400)
    assert not fires(calm)
    assert not fires(3.0 * rng.normal(size=400))  # 从预热起就持续的失配
    assert fires(np.r_[calm[:200], 3.0 * rng.normal(size=200)])  # 预热后出现的失配


def test_profile_handles_rank_deficient_designs() -> None:
    """恒定变量使持续性列与偏置列共线；RSS 仍是精确的最小二乘残差。"""

    rng = np.random.default_rng(3)
    mean, scale = rng.normal(size=(25, 1)), np.full((25, 1), 0.5)
    current = np.full((25, 1), 2.0)
    following = 0.7 * mean + 0.2 + 0.5 * rng.normal(size=(25, 1))
    profile = Profile((1,), 0.0)
    for t in range(25):
        profile.update(*features(mean[t], scale[t], following[t], current[t]), scale[t])
    phi, y = features(mean, scale, following, current)
    theta = np.linalg.lstsq(phi[:, 0], y[:, 0], rcond=None)[0]
    rss = np.sum((y[:, 0] - phi[:, 0] @ theta) ** 2)
    expected = -12.5 * np.log(2 * math.pi * math.e * rss / 25) - np.log(scale).sum()
    assert math.isclose(profile.log_likelihood()[0], expected, rel_tol=1e-9)
    assert Profile((1,), 0.1).log_likelihood()[0] == 0.0


def test_profile_variance_floor_bounds_constant_records() -> None:
    """恒定记录（RSS = 0）的上确界在 c^2 = c_min^2 处取到，有限。"""

    profile = Profile((1,), 0.1)
    for _ in range(10):
        profile.update(*features(np.ones(1), np.full(1, 2.0), np.ones(1), np.ones(1)), np.full(1, 2.0))
    expected = -5 * math.log(2 * math.pi * 0.01) - 10 * math.log(2.0)
    assert math.isclose(profile.log_likelihood()[0], expected, rel_tol=1e-9)


def test_computed_family_finds_the_source_the_record_is_computed_from() -> None:
    """记录由来源 2 算出时，计算族的边际集中在含 2 的支撑上，且远高于有自身新息的记录。"""

    from seer.methods.evidence import Computed

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(300, 4)), axis=0)
    computed = 0.7 * x[:, 2] + 0.3
    own = np.cumsum(rng.normal(size=300))  # 自身新息
    route = np.full((1, 2, 4), 0.25)
    route[0, :, 0] = 0.0  # 被检验变量 0 自身
    family = Computed(route, lags=2, alpha=2.0, beta=1.0, floor=0.1)
    visible = np.ones((2, 4), dtype=bool)
    visible[:, 0] = False
    for t in range(2, 300):
        levels = np.stack([x[t, :], x[t - 1, :]], axis=1)  # (d, L)：l = 0, 1
        family.update(levels, visible, np.array([computed[t], own[t]]), np.ones(2))
    singles = family.log_prior[0] + family.log_m[0]
    assert int(np.argmax(singles[0, 0])) == 2
    log_m = family.log_marginal()[0]
    assert log_m[0] - log_m[1] > 200


def test_computed_family_with_a_single_routed_source_is_finite() -> None:
    """只有一个来源有文本引用时，先验全部归单来源，边际有限。"""

    from seer.methods.evidence import Computed

    family = Computed(np.array([[[0.0, 1.0, 0.0]]]), lags=1, alpha=2.0, beta=1.0, floor=0.1)
    assert np.isfinite(family.log_marginal()).all()
    family.update(np.ones((3, 1)), np.ones((1, 3), dtype=bool), np.array([1.0]), np.ones(1))
    assert np.isfinite(family.log_marginal()).all()


def test_original_semantics_covers_reconstructions_as_exact_as_in_normal_operation() -> None:
    """训练中成立的近确定关系（记录 0 = 0.7 x_2 + 0.3 + 0.01 eps）在测试中保持时，零族的重构（方差不低于
    训练中的最小新息）支配候选的计算族；记录改由来源 1 以新系数算出（同样的噪声或精确）时，候选得到大量证据：
    单来源的自由系数不能表示来源替换，未确立的来源对不在零族中。"""

    from seer.methods.evidence import Computed, Natural, innovation_floors, supports

    rng = np.random.default_rng(0)
    T, d, lags = 600, 4, 2
    x = np.cumsum(rng.normal(size=(T, d)), axis=0)
    x[:, 0] = 0.7 * x[:, 2] + 0.3 + 0.01 * rng.normal(size=T)
    starts = np.zeros(T, dtype=int)
    rows = np.arange(2, 300)
    floors = innovation_floors(x, rows, starts, lags, np.array([0]), np.ones((len(rows), 1)), window=30)
    every = supports(d)
    assert np.isinf(floors[0, every.index((0,))]) and np.isinf(floors[0, every.index((0, 2))])
    assert floors[0, every.index((2,))] < 2e-4 < floors[0, every.index((1,))]  # 成立关系的新息即其噪声
    assert np.isinf(floors[0, every.index((1, 2))])  # 来源对未确立：其自由系数可表示以来源 1 替换来源 2
    visible = np.ones((1, d), dtype=bool)
    visible[:, 0] = False
    route = np.full((1, 1, d), 1 / 3)
    route[0, :, 0] = 0.0
    for case in ("kept", "noisy", "exact"):
        noise = {"kept": 0.0, "noisy": 0.01, "exact": 0.0}[case] * rng.normal(size=T)
        y = x[:, 0] if case == "kept" else 1.5 * x[:, 1] - 2.0 + noise
        candidate = Computed(route, lags=lags, alpha=2.0, beta=1.0, floor=1e-3)
        null = Natural(floors, np.array([0]), visible, lags, floor=1e-3)
        for t in range(300, T - 1):
            levels = np.stack([x[t + 1], x[t]], axis=1)
            candidate.update(levels, visible, y[t + 1 : t + 2], np.ones(1))
            if t == 330:  # 预热：候选在前 30 个样本上学习，零族只取其后的样本
                burned = candidate.log_marginal()[0, 0]
                candidate.restart()
            if t >= 330:
                null.update(levels, y[t + 1 : t + 2], np.ones(1))
        gain = candidate.log_marginal()[0, 0] - burned - null.log_likelihood()[0]
        assert gain < 0 if case == "kept" else gain > 200, (case, gain)
