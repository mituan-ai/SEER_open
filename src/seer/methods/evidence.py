"""两层语义信念的证据计算，只接收数值数组。

对语义 S 与观测变量 j，观测头给出 x_{j,t+1} 在 F_t 与同时刻已揭示记录下的 mu、sigma；标准化特征与观测为

    phi = [mu, x_{j,t}, 1] / sigma,    y = x_{j,t+1} / sigma .

原语义的数值适应族 F_0 是 y 对 phi 的线性高斯回归（系数 theta、方差 c^2 >= c_min^2），即网络预测与持续性预测的
任意仿射组合。所有似然都在原始观测空间比较：标准化空间的密度再乘 1 / sigma。
被检验集 U 上按变量序链式揭示（``seer.methods.seer.visibility``），各因子之积是 x_{U,t+1} 在 F_t 与 x_{-U,t+1}
下的条件联合密度；全部候选与原族在同一组数据、同一条件下比较。

- ``Prequential``：候选头族在以网络自身预测为中心（theta_0 = [1, 0, 0]）的正态–逆伽马先验下的前序预测，
  先预测后更新（秩一）。y 的量级为 1 / sigma，零均值先验会使前几步残差极大并长期抬高方差估计。
  预测尺度取 max(beta / alpha, c_min^2)：仍是归一化、可预测的 Student-t 密度（证据有效性只需此），
  但不是方差先验截断到 c^2 >= c_min^2 时的精确后验预测。
- ``Computed``：候选的来源重构族——记录由卡片文本引用的来源（单个或成对）的有限时滞仿射函数算出；
  支撑先验来自文本，参数在线估计；预热后以参考后验预测（``restart``），可表达精确计算。
  候选的边际取头族与重构族各 1/2 的混合（``seer.training.online``）。
- ``innovation_floors`` / ``Natural``：原语义的第二个零族——来源重构，系数自由取上确界，方差不低于训练中
  该支撑出现过的最小新息：原语义下没有记录会比正常运行中任何时候都更精确地由来源算出。单来源全部
  进入；来源对只在确立时进入（自由系数可表示来源替换）。
- ``Profile``：原族在事件内已实现时刻上的最大似然，不加惩罚；方差乘子取 max(RSS / n, c_min^2)，
  上确界处处有限（恒定记录使 RSS = 0 时也是）。零族逐变量取头族与天然重构的上确界之并。
- ``belief``：候选 k 只在其修订的观测变量 U_k 上检验，E^{(k)} = m_k(U_k) / L_0(U_k)；
  w / (1 - w) = pi / (1 - pi) sum_k rho_k E^{(k)}，份额 lambda_k ∝ rho_k E^{(k)} 是证据导出的选择权重
  （各候选检验的变量集可不同，不是同一结果上的后验）。数据来自零族中某个分布时，上确界不小于该分布的
  似然，E 受似然比鞅支配（e 过程，本身不一定是超鞅）。事件的前 W 个样本只用于学习
  （``seer.training.online.Event``）：候选以其上的后验为先验，零族的上确界只取其后的样本，以 F_W 为条件
  仍是 e 过程。
- ``event_mass``：多事件风险预算 epsilon_e = epsilon / (e (e + 1))，反解每个事件的 pi_e。
- ``Monitor``：触发统计量。原语义观测头经稳健的在线位置校准后的截断平方残差，以训练与预热段的均值为
  水平归一化，单边 CUSUM；只决定何时提案，不进入证据。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

DIM = 3  # 原族的校准维数：网络预测增益、持续性增益与偏置
CENTER = np.array([1.0, 0.0, 0.0])  # 先验中心：网络自身的预测
gammaln = np.vectorize(math.lgamma, otypes=[float])


def logsumexp(a: np.ndarray) -> float:
    top = float(np.max(a))
    return top if math.isinf(top) else top + math.log(float(np.sum(np.exp(a - top))))


def features(mean, scale, following, current) -> tuple[np.ndarray, np.ndarray]:
    """观测头输出、下一步与当前记录 (..., J) -> phi (..., J, 3) 与 y (..., J)。"""

    current, following = np.broadcast_to(current, mean.shape), np.broadcast_to(following, mean.shape)
    phi = np.stack([mean, current, np.ones_like(mean)], axis=-1) / scale[..., None]
    return phi, following / scale


class Prequential:
    """形状为 ``shape``（如 候选 x 变量）的一组独立贝叶斯线性回归，先验均值 ``center``（维数即其长度）。

    后验协方差按 Sherman–Morrison 秩一更新，与逐步求解精度矩阵相同。
    """

    def __init__(
        self, shape: tuple[int, ...], alpha: float, beta: float, floor: float, center: np.ndarray = CENTER
    ) -> None:
        center = np.asarray(center, dtype=float)
        self.covariance = np.broadcast_to(np.eye(len(center)), (*shape, len(center), len(center))).copy()
        self.mean = np.broadcast_to(center, (*shape, len(center))).copy()
        self.alpha = np.full(shape, float(alpha))
        self.beta = np.full(shape, float(beta))
        self.floor = float(floor) ** 2

    def update(self, phi: np.ndarray, y: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回原始空间的对数预测密度与单位方差的标准化残差，再以 (phi, y) 更新后验。"""

        solved = np.einsum("...ij,...j->...i", self.covariance, phi)
        ell = 1.0 + np.maximum(np.sum(phi * solved, axis=-1), 0.0)  # 协方差半正定；防数值误差
        residual = y - np.sum(phi * self.mean, axis=-1)
        nu = 2.0 * self.alpha
        square = np.maximum(self.beta / self.alpha, self.floor) * ell  # 噪声方差的定义域 c^2 >= c_min^2
        log_density = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * math.pi * square)
            - (nu + 1) / 2 * np.log1p(residual**2 / (nu * square))
            - np.log(scale)
        )
        z = residual / np.sqrt(square * nu / (nu - 2))
        self.covariance = self.covariance - solved[..., :, None] * solved[..., None, :] / ell[..., None, None]
        self.covariance = 0.5 * (self.covariance + np.swapaxes(self.covariance, -1, -2))
        self.mean = self.mean + solved * (residual / ell)[..., None]
        self.alpha = self.alpha + 0.5
        self.beta = self.beta + residual**2 / (2 * ell)
        return log_density, z


def reconstruction_inputs(levels: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """重构族的输入 (J, d, L)：被检验变量 j 可用的来源记录，l = 0 不可见者取零；坐标取 x_{s,t+1} 与
    相邻时滞之差（与水平张成同一空间，相邻水平几乎相同时仍良态；恒定来源的差恒为零）。"""

    lagged = np.broadcast_to(levels, (len(visible), *levels.shape)).copy()
    lagged[:, :, 0] *= visible
    lagged[:, :, 1:] = lagged[:, :, :-1] - lagged[:, :, 1:]
    return lagged


class Computed:
    """候选语义的来源重构族：被检验变量 j 的记录由来源集合 A 算出，

        x_{j,t+1} = sum_{s in A} sum_{l < L} a_{sl} x_{s,t+1-l} + b + c eps,

    参数以零为中心的正态–逆伽马前序估计（坐标取 x_{s,t+1} 与相邻时滞之差，与水平张成同一空间），
    不读原语义的网络预测与 j 自身的历史。
    支撑 A 取单个来源或来源对，先验各 1/2，组内正比于卡片文本引用 pi 之积（只有一个可用来源时全部
    归单来源）：语义给出结构的先验，数据估计参数。自然测量的记录有自身的新息，只靠其他记录的水平
    无法胜过原语义的一步预测；记录确由来源算出时此族精确。各支撑的回归与卡片无关，所有候选共用，
    候选只在支撑先验上不同。同时刻（l = 0）不可见的来源（链式顺序中尚未揭示者）其 l = 0 特征取零。

    ``route`` (K, J, d)：各候选、各被检验变量的文本引用（自身为零）；``log_marginal`` 给出 (K, J)。
    """

    def __init__(self, route: np.ndarray, lags: int, alpha: float, beta: float, floor: float) -> None:
        K, J, d = route.shape
        self.lags = int(lags)
        pairs = np.array([(a, b) for a in range(d) for b in range(a + 1, d)], dtype=int).reshape(-1, 2)
        self.supports = (np.arange(d)[:, None], pairs)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pi = np.log(route / route.sum(axis=-1, keepdims=True))
            groups = [log_pi[..., sup].sum(axis=-1) for sup in self.supports]  # 组内未归一化
            norms = [np.logaddexp.reduce(g, axis=-1, initial=-np.inf)[..., None] for g in groups]
            usable = np.isfinite(norms[1])  # 至少两个来源有文本引用
            share = np.where(usable, math.log(0.5), 0.0)
            single = share + groups[0] - norms[0]
            pair = np.where(usable, share + groups[1] - norms[1], -np.inf)
        self.log_prior = [single, pair]
        if not np.isfinite(norms[0]).all():
            raise ValueError("every tested variable needs at least one routed source")
        self.models = [
            Prequential((J, len(sup)), alpha, beta, floor, np.zeros(self.lags * sup.shape[1] + 1))
            for sup in self.supports
        ]
        self.log_m = [np.zeros((J, len(sup))) for sup in self.supports]
        self.grams: list[np.ndarray] | None = [  # 预热中累积的 [phi, y]^T [phi, y]
            np.zeros((J, len(sup), self.lags * sup.shape[1] + 2, self.lags * sup.shape[1] + 2))
            for sup in self.supports
        ]

    def update(self, levels: np.ndarray, visible: np.ndarray, y: np.ndarray, scale: np.ndarray) -> None:
        """levels (d, L)：各变量 x_{s,t+1-l}；visible (J, d)：同时刻可见性；y、scale (J,)：被检验变量的
        下一步记录与原语义尺度。"""

        J = len(y)
        lagged = reconstruction_inputs(levels, visible)
        for g, sup in enumerate(self.supports):
            x = lagged[:, sup].reshape(J, len(sup), -1)  # (J, 支撑数, |A| L)
            phi = np.concatenate([x, np.ones((J, len(sup), 1))], axis=-1) / scale[:, None, None]
            shape = (J, len(sup))
            target = np.broadcast_to((y / scale)[:, None], shape)
            log_density, _ = self.models[g].update(phi, target, np.broadcast_to(scale[:, None], shape))
            self.log_m[g] += log_density
            if self.grams is not None:
                row = np.concatenate([phi, target[..., None]], axis=-1)
                self.grams[g] += row[..., :, None] * row[..., None, :]

    def restart(self) -> None:
        """预热结束：各支撑的预测改用参考先验（p(theta, c^2) ∝ 1 / c^2）在预热样本上的后验，
        theta ~ N(theta_hat, c^2 (X^T X)^-1)、c^2 ~ IG(alpha_W, RSS / 2)，不带系数先验与噪声尺度先验的代价。
        X^T X 加相对其迹 1e-9 的岭以保持正定（秩亏时，例如恒定来源）。支撑的混合权重仍是正常先验下预热的
        后验（只依赖 F_W）；精确算出的记录 RSS = 0，预测尺度即刻落到下界。"""

        if self.grams is None:
            return
        for model, gram in zip(self.models, self.grams, strict=True):
            g, b = gram[..., :-1, :-1], gram[..., :-1, -1]
            p = g.shape[-1]
            ridge = 1e-9 * np.trace(g, axis1=-2, axis2=-1)[..., None, None] / p + 1e-12
            covariance = np.linalg.inv(g + ridge * np.eye(p))
            covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
            mean = np.einsum("...ij,...j->...i", covariance, b)
            model.mean, model.covariance = mean, covariance
            model.beta = np.maximum(gram[..., -1, -1] - np.sum(mean * b, axis=-1), 0.0) / 2.0
        self.grams = None

    def log_marginal(self) -> np.ndarray:
        joint = np.concatenate([p + m for p, m in zip(self.log_prior, self.log_m, strict=True)], axis=-1)
        return np.logaddexp.reduce(joint, axis=-1)


def source_levels(values: np.ndarray, rows: np.ndarray, starts: np.ndarray, lags: int) -> np.ndarray:
    """(n, d, L)：转移 t -> t + 1 的来源记录 x_{s,t+1-l}，段首之前取段首。"""

    index = np.maximum(rows[:, None] + 1 - np.arange(lags), starts[rows][:, None])
    return values[index].transpose(0, 2, 1)


def supports(d: int) -> list[tuple[int, ...]]:
    """重构族的全部支撑：单来源与来源对（与 ``Computed`` 同序）。"""

    return [(s,) for s in range(d)] + list(itertools.combinations(range(d), 2))


def _theta(grams: np.ndarray, cols: list[int]) -> np.ndarray:
    """各块 Gram 矩阵 [z, y]^T [z, y] 上 y 对 z[cols] 的最小二乘系数（最后一列为 y）。"""

    inverse = np.linalg.pinv(grams[:, cols][:, :, cols], hermitian=True)
    return np.einsum("nij,nj->ni", inverse, grams[:, cols, -1])


def _rss(grams: np.ndarray, cols: list[int], theta: np.ndarray) -> np.ndarray:
    """系数 theta 在各块上的残差平方和（截于零）。"""

    g = grams[:, cols][:, :, cols]
    fit = -2 * np.einsum("ni,ni->n", theta, grams[:, cols, -1]) + np.einsum("ni,nij,nj->n", theta, g, theta)
    return np.maximum(grams[:, -1, -1] + fit, 0.0)


def innovation_floors(
    values: np.ndarray,
    rows: np.ndarray,
    starts: np.ndarray,
    lags: int,
    observed: np.ndarray,
    scale: np.ndarray,
    window: int,
) -> np.ndarray:
    """原语义下各记录相对来源重构的新息下界 v (J, 支撑数)，sigma 单位；不属于原语义的支撑记为 inf。

    训练转移按段内连续的 ``window`` 个切块，块内以普通最小二乘重构 x_{j,t+1}（特征与 ``Computed`` 张成
    同一空间，完整同时刻信息），v_{jA} 取各块无偏残差方差 RSS / (n - p) 的最小值：正常运行中某处出现过的
    最精确程度。单来源支撑的自由系数只重标定同一来源，全部属于原语义；来源对嵌套其单来源，自由系数可表示
    以一个来源替换另一个，只有确立的联合关系属于原语义：相邻块中前块拟合、后块检验（样本外），至少一半
    块中残差平方和小于其每个真子支撑（空支撑即 j 自身持续性 [x_{j,t}, 1]）。含 j 自身的支撑不进入零族。
    ``scale`` (n, J)：原语义观测头在这些转移上的 sigma。
    """

    n, d = len(rows), values.shape[1]
    D = d * lags
    z = np.concatenate([source_levels(values, rows, starts, lags).reshape(n, -1), np.ones((n, 1))], axis=1)
    cuts = np.flatnonzero((np.diff(rows) != 1) | (starts[rows][1:] != starts[rows][:-1])) + 1
    blocks, adjacent = [], []  # 段内连续转移切成长度 window 的块；adjacent：同一段内相邻块的前块
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, n], strict=True):
        first = len(blocks)
        blocks += [(i, i + window) for i in range(a, b + 1 - window, window)]
        adjacent += range(first, len(blocks) - 1)
    adjacent = np.asarray(adjacent, dtype=int)
    every = supports(d)
    floors = np.full((len(observed), len(every)), np.inf)
    for o, j in enumerate(observed):
        w = np.concatenate([z, values[rows + 1, j][:, None]], axis=1) / scale[:, o, None]
        grams = np.stack([w[i:k].T @ w[i:k] for i, k in blocks])
        fit, test = grams[adjacent], grams[adjacent + 1]

        def columns(sup: tuple[int, ...]) -> list[int]:
            return [s * lags + k for s in sup for k in range(lags)] + [D]

        rss = {(): _rss(test, [int(j) * lags + 1, D], _theta(fit, [int(j) * lags + 1, D]))}
        for c, sup in enumerate(every):
            if j in sup:
                continue
            if len(sup) == 2:
                for part in [sup, *((s,) for s in sup)]:
                    if part not in rss:
                        rss[part] = _rss(test, columns(part), _theta(fit, columns(part)))
                if np.mean(rss[sup] < np.min([rss[(sup[0],)], rss[(sup[1],)], rss[()]], axis=0)) < 0.5:
                    continue
            cols = columns(sup)
            floors[o, c] = float(np.min(_rss(grams, cols, _theta(grams, cols)) / max(window - len(cols), 1)))
    return floors


class Natural:
    """原语义的数值现实：每条记录相对任何来源重构的新息不低于它在正常运行中的最小值（``innovation_floors``）。

    零族在全部支撑上取重构族（同 ``Computed`` 的特征）的轮廓上确界，系数自由，方差 c^2 >= v_{jA}：
    关系的局部重标定、正常运行中出现过的近确定关系（控制器输出跟随被控量、保持与量化等）都属于原语义；
    候选只有声称记录比正常运行中任何时候都更精确地由来源算出时才可能得到证据。可见性只限制零族在事件中
    可用的坐标（同时刻不可见的来源只取其时滞），与 ``Computed`` 相同；下界以完整信息标定，因而偏低、
    零族偏大。

    ``floors`` (J_all, 支撑数)；``tested``：被检验变量在被观测变量中的序号；``visible`` (J, d)；
    ``floor``：计算族的尺度下界，零族的方差取 max(v, floor^2)（训练中出现过的恒定或插值使 v = 0 时
    上确界仍有限，与候选同一下界、不产生证据）。``log_likelihood`` 给出 (J,)。
    """

    def __init__(
        self, floors: np.ndarray, tested: np.ndarray, visible: np.ndarray, lags: int, floor: float
    ) -> None:
        D = visible.shape[1] * lags
        groups: dict[int, list[tuple[int, list[int], float]]] = {}
        for t, o in enumerate(tested):
            for c, sup in enumerate(supports(visible.shape[1])):
                if np.isfinite(floors[o, c]):
                    cols = [s * lags + k for s in sup for k in range(lags) if k > 0 or visible[t, s]] + [D]
                    groups.setdefault(len(cols), []).append((t, cols, floors[o, c]))
        self.items = [
            (
                np.array([t for t, _, _ in g]),
                np.array([c for _, c, _ in g]),
                Profile((len(g),), np.sqrt(np.maximum([v for _, _, v in g], floor**2)), n),
            )
            for n, g in groups.items()
        ]
        self.J = len(tested)

    def update(self, levels: np.ndarray, y: np.ndarray, scale: np.ndarray) -> None:
        """levels (d, L)：x_{s,t+1-l}；y、scale (J,)：被检验变量的下一步记录与原语义尺度。"""

        f = np.append(levels.reshape(-1), 1.0)
        for index, cols, profile in self.items:
            sigma = scale[index]
            profile.update(f[cols] / sigma[:, None], y[index] / sigma, sigma)

    def reset(self) -> None:
        for _, _, profile in self.items:
            profile.reset()

    def log_likelihood(self) -> np.ndarray:
        best = np.full(self.J, -np.inf)
        for index, _, profile in self.items:
            np.maximum.at(best, index, profile.log_likelihood())
        return best


class Profile:
    """形状为 ``shape`` 的一组原族最大似然；累积 [phi, y] 的 QR 三角因子 R = [R_phi, r_y]，
    RSS = min_theta ||r_y - R_phi theta||^2（秩亏时仍精确，例如恒定变量使持续性列与偏置列共线）。"""

    def __init__(self, shape: tuple[int, ...], floor: float | np.ndarray, dim: int = DIM) -> None:
        self.dim = int(dim)
        self.r = np.zeros((*shape, self.dim + 1, self.dim + 1))
        self.log_scale = np.zeros(shape)
        self.floor = np.asarray(floor, dtype=float) ** 2
        self.n = 0

    def reset(self) -> None:
        self.r[:] = 0.0
        self.log_scale[:] = 0.0
        self.n = 0

    def update(self, phi: np.ndarray, y: np.ndarray, scale: np.ndarray) -> None:
        row = np.concatenate([phi, y[..., None]], axis=-1)[..., None, :]
        self.r = np.linalg.qr(np.concatenate([self.r, row], axis=-2), mode="r")
        self.log_scale += np.log(scale)
        self.n += 1

    def log_likelihood(self) -> np.ndarray:
        """sup_{theta, c^2 >= c_min^2} 的对数似然：-n/2 log(2 pi v) - RSS / (2 v) - sum log sigma，
        v = max(RSS / n, c_min^2)。"""

        if self.n == 0:
            return np.zeros(self.log_scale.shape)
        design, response = self.r[..., : self.dim], self.r[..., self.dim]
        theta = np.einsum("...ij,...j->...i", np.linalg.pinv(design), response)
        rss = np.sum((response - np.einsum("...ij,...j->...i", design, theta)) ** 2, axis=-1)
        v = np.maximum(rss / self.n, self.floor)
        return -0.5 * self.n * np.log(2 * math.pi * v) - rss / (2 * v) - self.log_scale


@dataclass(frozen=True)
class Belief:
    w: float  # 修订权重 1 - q(S_0)
    shares: np.ndarray  # lambda_k，(K,)
    log_e: float  # log E_t
    log_e_each: np.ndarray  # log E_t^{(k)}，(K,)

    def weights(self) -> np.ndarray:
        """q_t(S_0), q_t(S_1), ..., q_t(S_K)。"""

        return np.concatenate([[1.0 - self.w], self.w * self.shares])


def belief(
    log_m: np.ndarray, log_profile: np.ndarray, tested: np.ndarray, rho: np.ndarray, pi: float
) -> Belief:
    """由候选逐变量前序对数似然 (K, J)、原族逐变量轮廓对数似然 (J,) 与各候选的被检验集 U_k
    （bool (J,) 或 (K, J)）得到两层权重：E^{(k)} = m_k(U_k) / L_0(U_k)，E = sum_k rho_k E^{(k)}，
    lambda_k ∝ rho_k E^{(k)}。只取链式条件因子的子集，原语义下仍是似然比鞅（被上确界支配）。"""

    mask = np.broadcast_to(tested, log_m.shape)
    each = np.where(mask, log_m - log_profile, 0.0).sum(axis=1)
    joint = np.log(rho) + each
    log_e = logsumexp(joint)
    log_odds = math.log(pi / (1 - pi)) + log_e
    w = 1.0 / (1.0 + math.exp(-log_odds)) if log_odds > 0 else math.exp(log_odds) / (1.0 + math.exp(log_odds))
    return Belief(w, np.exp(joint - log_e), log_e, each)


def risk(pi: float, tau: float) -> float:
    """Ville 界：原语义成立时 P(存在 t: w_t >= tau) <= pi (1 - tau) / ((1 - pi) tau)。"""

    return pi * (1 - tau) / ((1 - pi) * tau)


def event_mass(budget: float, tau: float, index: int) -> tuple[float, float]:
    """第 e 个事件（从 1 起）的风险 epsilon_e = budget / (e (e + 1)) 与对应的 pi_e。

    sum_e epsilon_e = budget：一个测试流上任一事件错误越界的概率不超过 budget。
    """

    epsilon = budget / (index * (index + 1))
    return epsilon, epsilon * tau / (1 - tau + epsilon * tau)


class Monitor:
    """当前语义的失配监控（触发事件）。

    位置以 [mu, x, 1] 在线校准（正态–逆伽马后验均值，同原族；以截尾于 +-cap 的残差更新，Huber 型），
    残差 r 以观测头的尺度 sigma 计量：
    残差重尾时高斯方差的在线估计被尖峰主导，以它标准化会淹没一切失配。r^2 截断于 cap^2，按参照水平
    归一化后进入单边 CUSUM

        C <- max(0, C + min(r^2 / level, cap) - 1 - drift)，

    任一变量超过它自己的阈值（逐变量标定，见 ``cusum_threshold``）即越界。参照水平取正常运行流上的
    水平与预热 W 个样本上的水平中的较大者：当前语义自身持续的失配（例如批准后网络仍预测不好的记录）
    是它的常态，此后新出现的失配才越界。只在语义改变（起始与批准）时重新预热；事件打开不改变语义，
    只清零 CUSUM 并经过 W 个样本的不应期（``refract``），未解决的失配不会被当作常态。``local`` 为 False
    时只用参照水平（用于标定）。
    """

    def __init__(
        self,
        level: np.ndarray,
        warmup: int,
        alpha: float,
        beta: float,
        floor: float,
        cap: float,
        drift: float,
        threshold: float | np.ndarray,
        local: bool = True,
    ) -> None:
        self.calibration = Prequential(level.shape, alpha, beta, floor)
        self.level, self.local = np.maximum(np.asarray(level, dtype=float), floor**2), local
        self.warmup, self.cap = int(warmup), float(cap)
        self.drift, self.threshold = float(drift), np.asarray(threshold, dtype=float)
        self.value = np.zeros(level.shape)
        self.seen = np.zeros(level.shape)
        self.count = 0
        self.quiet = 0

    def refract(self) -> None:
        self.value[:] = 0.0
        self.quiet = self.warmup

    def residual(self, phi: np.ndarray, y: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """以 sigma 计量的校准残差平方（截断于 cap^2），再以 (phi, y) 更新校准。"""

        r = y - np.sum(phi * self.calibration.mean, axis=-1)
        self.calibration.update(phi, y - r + np.clip(r, -self.cap, self.cap), scale)  # 尖峰不移动校准
        return np.minimum(r**2, self.cap**2)

    def update(self, phi: np.ndarray, y: np.ndarray, scale: np.ndarray) -> bool:
        r2 = self.residual(phi, y, scale)
        self.count += 1
        if self.count <= self.warmup:
            self.seen += r2
            if self.count == self.warmup and self.local:
                self.level = np.maximum(self.level, self.seen / self.warmup)
            return False
        if self.quiet > 0:
            self.quiet -= 1
            return False
        self.value = np.maximum(0.0, self.value + np.minimum(r2 / self.level, self.cap) - 1.0 - self.drift)
        return bool((self.value > self.threshold).any())


