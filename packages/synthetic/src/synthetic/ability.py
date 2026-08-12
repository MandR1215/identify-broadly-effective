from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import optimize, stats


class AbilityDistribution(Protocol):
    def sample(
        self,
        size: int | tuple[int, int],
        random_state: np.random.Generator | int | None = None,
    ) -> np.ndarray: ...

    def mean(self) -> float: ...

    def cvar(self, gamma: float) -> float: ...


def _validate_gamma(gamma: float) -> None:
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must satisfy 0 < gamma <= 1")


def _to_rng(
    random_state: np.random.Generator | int | None,
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state

    return np.random.default_rng(random_state)


def _truncnorm_component_mean(
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    Z = stats.norm.cdf(b) - stats.norm.cdf(a)
    return mu + sigma * (stats.norm.pdf(a) - stats.norm.pdf(b)) / Z


def _truncnorm_lower_partial_first_moment(
    upper: float,
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Returns E[X * 1{X <= upper}] for each truncated normal component.
    """

    z = (upper - mu) / sigma
    z = np.clip(z, a, b)

    Z = stats.norm.cdf(b) - stats.norm.cdf(a)

    return (
        mu * (stats.norm.cdf(z) - stats.norm.cdf(a))
        + sigma * (stats.norm.pdf(a) - stats.norm.pdf(z))
    ) / Z


@dataclass
class TruncatedNormal:
    mu: float
    sigma: float
    trunc_left: float
    trunc_right: float
    a: float = field(init=False)
    b: float = field(init=False)

    def __post_init__(self):
        assert self.sigma > 0
        self.a = (self.trunc_left - self.mu) / self.sigma
        self.b = (self.trunc_right - self.mu) / self.sigma

    def sample(
        self,
        size: int | tuple[int,int],
        random_state: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        return stats.truncnorm.rvs(
            a=self.a,
            b=self.b,
            loc=self.mu,
            scale=self.sigma,
            size=size,
            random_state=_to_rng(random_state),
        )
    
    
    def mean(self) -> float:
        return float(_truncnorm_component_mean(a=np.asarray(self.a), b=np.asarray(self.b), mu=np.asarray(self.mu), sigma=np.asarray(self.sigma)))
        
    
    def cvar(self, gamma: float) -> float:
        _validate_gamma(gamma=gamma)
        if gamma == 1.0: 
            return self.mean()
        z_gamma = stats.truncnorm.ppf(gamma, a=self.a, b=self.b, loc=0.0, scale=1.0)
        Z = stats.norm.cdf(self.b) - stats.norm.cdf(self.a)
        numerator = (self.mu * gamma + self.sigma * (stats.norm.pdf(self.a) - stats.norm.pdf(z_gamma)) / Z)
        return float(numerator / gamma)
    

@dataclass
class TruncatedNormalMixture:
    weights: np.ndarray
    mus: np.ndarray
    sigmas: np.ndarray
    trunc_lefts: np.ndarray
    trunc_rights: np.ndarray
    a: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        mus = np.asarray(self.mus, dtype=float)
        sigmas = np.asarray(self.sigmas, dtype=float)
        trunc_lefts = np.asarray(self.trunc_lefts, dtype=float)
        trunc_rights = np.asarray(self.trunc_rights, dtype=float)

        assert len(weights) > 0 and weights.ndim == 1
        assert np.all(weights > 0.0)
        assert np.isclose(weights.sum(), 1.0)
        assert weights.shape == mus.shape == sigmas.shape == trunc_lefts.shape == trunc_rights.shape
        assert np.all(sigmas > 0.0)
        assert np.all(trunc_lefts < trunc_rights)

        self.weights = weights
        self.mus = mus
        self.sigmas = sigmas
        self.trunc_lefts = trunc_lefts
        self.trunc_rights = trunc_rights

        self.a = (self.trunc_lefts - self.mus) / self.sigmas
        self.b = (self.trunc_rights - self.mus) / self.sigmas

    @property
    def support_left(self) -> float:
        return float(np.min(self.trunc_lefts))

    @property
    def support_right(self) -> float:
        return float(np.max(self.trunc_rights))

    def mean(self) -> float:
        component_means = _truncnorm_component_mean(a=self.a, b=self.b, mu=self.mus, sigma=self.sigmas)
        return float(np.sum(self.weights * component_means))

    def sample(
        self,
        size: int | tuple[int, int],
        random_state: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        rng = _to_rng(random_state)

        component_ids = rng.choice(len(self.weights), size=size, p=self.weights)

        abilities = stats.truncnorm.rvs(
            a=self.a[component_ids],
            b=self.b[component_ids],
            loc=self.mus[component_ids],
            scale=self.sigmas[component_ids],
            random_state=rng,
        )

        return np.asarray(abilities, dtype=float)

    def cdf(self, x: float) -> float:
        component_cdfs = stats.truncnorm.cdf(
            x,
            a=self.a,
            b=self.b,
            loc=self.mus,
            scale=self.sigmas,
        )
        return float(np.sum(self.weights * component_cdfs))

    def ppf(self, gamma: float) -> float:
        _validate_gamma(gamma)
        if gamma == 1.0: 
            return self.support_right

        return float(
            optimize.brentq(
                lambda x: self.cdf(x) - gamma,
                self.support_left,
                self.support_right,
            )
        )

    def cvar(self, gamma: float) -> float:
        _validate_gamma(gamma)
        if gamma == 1.0: 
            return self.mean()

        q_gamma = self.ppf(gamma)

        component_partial_moments = _truncnorm_lower_partial_first_moment(
            upper=q_gamma,
            a=self.a,
            b=self.b,
            mu=self.mus,
            sigma=self.sigmas,
        )

        partial_moment = np.sum(self.weights * component_partial_moments)

        return float(partial_moment / gamma)