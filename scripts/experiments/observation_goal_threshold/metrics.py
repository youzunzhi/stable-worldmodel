"""Deterministic threshold selection and held-out metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import LABEL_F, LABEL_T, LABEL_U


class CalibrationOutcome(RuntimeError):
    """A registered scientific stop, rather than an implementation failure."""

    def __init__(self, code: str, details: dict | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


def _adjusted_macro_weights(
    labels: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    target_label: np.uint8,
) -> np.ndarray:
    selected = labels == target_label
    target_groups, inverse = np.unique(groups[selected], return_inverse=True)
    if not len(target_groups):
        raise ValueError(f'no records for label {int(target_label)}')
    totals = np.bincount(inverse, weights=weights[selected])
    if np.any(totals <= 0):
        raise ValueError('non-positive per-group design-weight total')
    adjusted = np.zeros(len(labels), dtype=np.float64)
    adjusted[selected] = (
        weights[selected] / totals[inverse] / len(target_groups)
    )
    return adjusted


def macro_curve(
    distance: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    anchor_group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact anchor-group-macro TPR/FPR curve over all unique distances."""
    distance = np.asarray(distance, dtype=np.float32)
    label = np.asarray(label, dtype=np.uint8)
    weight = np.asarray(weight, dtype=np.float64)
    group = np.asarray(anchor_group, dtype=np.int64)
    keep = (label == LABEL_T) | (label == LABEL_F)
    distance, label, weight, group = (
        value[keep] for value in (distance, label, weight, group)
    )
    t_weight = _adjusted_macro_weights(label, weight, group, LABEL_T)
    f_weight = _adjusted_macro_weights(label, weight, group, LABEL_F)
    order = np.argsort(distance, kind='stable')
    ordered_distance = distance[order]
    cumulative_t = np.cumsum(t_weight[order])
    cumulative_f = np.cumsum(f_weight[order])
    ends = np.r_[
        np.flatnonzero(ordered_distance[1:] != ordered_distance[:-1]),
        len(ordered_distance) - 1,
    ]
    return {
        'epsilon': ordered_distance[ends],
        'macro_tpr': cumulative_t[ends],
        'macro_fpr': cumulative_f[ends],
    }


def uniform_curve(
    distance: np.ndarray, label: np.ndarray, weight: np.ndarray
) -> dict[str, np.ndarray]:
    """Pair-population TPR/FPR/precision curve from the uniform sample."""
    distance = np.asarray(distance, dtype=np.float32)
    label = np.asarray(label, dtype=np.uint8)
    weight = np.asarray(weight, dtype=np.float64)
    keep = (label == LABEL_T) | (label == LABEL_F)
    distance, label, weight = (
        value[keep] for value in (distance, label, weight)
    )
    total_t = weight[label == LABEL_T].sum()
    total_f = weight[label == LABEL_F].sum()
    if total_t <= 0 or total_f <= 0:
        raise ValueError('uniform sample must contain both T and F')
    order = np.argsort(distance, kind='stable')
    d = distance[order]
    y = label[order]
    w = weight[order]
    cum_t = np.cumsum(w * (y == LABEL_T))
    cum_f = np.cumsum(w * (y == LABEL_F))
    ends = np.r_[np.flatnonzero(d[1:] != d[:-1]), len(d) - 1]
    tp, fp = cum_t[ends], cum_f[ends]
    denominator = tp + fp
    precision = np.divide(
        tp,
        denominator,
        out=np.ones_like(tp, dtype=np.float64),
        where=denominator > 0,
    )
    return {
        'epsilon': d[ends],
        'uniform_tpr': tp / total_t,
        'uniform_fpr': fp / total_f,
        'population_precision': precision,
    }


def _at_candidates(
    source_epsilon: np.ndarray,
    source_value: np.ndarray,
    candidates: np.ndarray,
    initial: float,
) -> np.ndarray:
    index = np.searchsorted(source_epsilon, candidates, side='right') - 1
    result = np.full(len(candidates), initial, dtype=np.float64)
    valid = index >= 0
    result[valid] = source_value[index[valid]]
    return result


@dataclass(frozen=True)
class SelectionResult:
    epsilon: float
    macro_tpr: float
    macro_fpr: float
    population_precision: float
    candidates: dict[str, np.ndarray]


def best_stratified_operating_point(
    stratified: dict[str, np.ndarray],
    *,
    max_negative_fpr: float,
) -> SelectionResult:
    """Return the registered best macro operating point without a recall gate.

    This is the complete threshold selector when population precision is not a
    constraint.  Keeping the recall gate outside this helper lets sensitivity
    reports retain Cube's best descriptive epsilon without promoting it.
    """
    macro = macro_curve(
        stratified['latent_distance'],
        stratified['label'],
        stratified['analysis_weight'],
        stratified['anchor_group'],
    )
    feasible = np.isfinite(macro['epsilon'])
    feasible &= macro['macro_fpr'] <= max_negative_fpr
    if not np.any(feasible):
        raise CalibrationOutcome(
            'THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT',
            {'reason': 'no finite candidate satisfies the FPR constraint'},
        )
    maximum_tpr = float(macro['macro_tpr'][feasible].max())
    winner = np.flatnonzero(feasible & (macro['macro_tpr'] == maximum_tpr))[0]
    candidates = {
        'epsilon': macro['epsilon'],
        'macro_tpr': macro['macro_tpr'],
        'macro_fpr': macro['macro_fpr'],
        'population_precision': np.full(
            len(macro['epsilon']), np.nan, dtype=np.float64
        ),
        'feasible': feasible,
    }
    return SelectionResult(
        epsilon=float(macro['epsilon'][winner]),
        macro_tpr=float(macro['macro_tpr'][winner]),
        macro_fpr=float(macro['macro_fpr'][winner]),
        population_precision=float('nan'),
        candidates=candidates,
    )


def select_threshold(
    stratified: dict[str, np.ndarray],
    uniform: dict[str, np.ndarray],
    *,
    min_positive_recall: float,
    max_negative_fpr: float,
    min_population_precision: float | None,
) -> SelectionResult:
    """Apply the registered max-macro-TPR/smallest-epsilon selector."""
    macro = macro_curve(
        stratified['latent_distance'],
        stratified['label'],
        stratified['analysis_weight'],
        stratified['anchor_group'],
    )
    population = uniform_curve(
        uniform['latent_distance'],
        uniform['label'],
        uniform['analysis_weight'],
    )
    candidates = np.union1d(macro['epsilon'], population['epsilon'])
    candidates = np.r_[np.float32(-np.inf), candidates, np.float32(np.inf)]
    tpr = _at_candidates(macro['epsilon'], macro['macro_tpr'], candidates, 0.0)
    fpr = _at_candidates(macro['epsilon'], macro['macro_fpr'], candidates, 0.0)
    precision = _at_candidates(
        population['epsilon'],
        population['population_precision'],
        candidates,
        1.0,
    )
    feasible = fpr <= max_negative_fpr
    if min_population_precision is not None:
        feasible &= precision >= min_population_precision
    feasible &= np.isfinite(candidates)
    if not np.any(feasible):
        raise CalibrationOutcome(
            'THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT',
            {'reason': 'no candidate satisfies FPR/precision constraints'},
        )
    maximum_tpr = float(tpr[feasible].max())
    if maximum_tpr < min_positive_recall:
        raise CalibrationOutcome(
            'THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT',
            {
                'best_macro_tpr': maximum_tpr,
                'required_macro_tpr': min_positive_recall,
            },
        )
    winner = np.flatnonzero(feasible & (tpr == maximum_tpr))[0]
    table = {
        'epsilon': candidates,
        'macro_tpr': tpr,
        'macro_fpr': fpr,
        'population_precision': precision,
        'feasible': feasible,
    }
    return SelectionResult(
        epsilon=float(candidates[winner]),
        macro_tpr=float(tpr[winner]),
        macro_fpr=float(fpr[winner]),
        population_precision=float(precision[winner]),
        candidates=table,
    )


def _weighted_ratio(
    mask: np.ndarray, denominator: np.ndarray, weight: np.ndarray
) -> float:
    den = float(weight[denominator].sum())
    return (
        float(weight[mask & denominator].sum() / den) if den else float('nan')
    )


def metrics_at_threshold(
    stratified: dict[str, np.ndarray],
    uniform: dict[str, np.ndarray],
    epsilon: float,
) -> dict:
    """Compute primary macro and secondary population metrics at epsilon."""
    sy = np.asarray(stratified['label'])
    sd = np.asarray(stratified['latent_distance'])
    sw = np.asarray(stratified['analysis_weight'], dtype=np.float64)
    sg = np.asarray(stratified['anchor_group'])
    inside_s = sd <= epsilon

    _, group_inverse = np.unique(sg, return_inverse=True)
    group_count = int(group_inverse.max()) + 1
    group_rates = []
    for target in (LABEL_T, LABEL_F):
        selected = sy == target
        totals = np.bincount(
            group_inverse,
            weights=sw * selected,
            minlength=group_count,
        )
        hits = np.bincount(
            group_inverse,
            weights=sw * selected * inside_s,
            minlength=group_count,
        )
        valid = totals > 0
        group_rates.append(hits[valid] / totals[valid])
    group_tpr, group_fpr = group_rates

    uy = np.asarray(uniform['label'])
    ud = np.asarray(uniform['latent_distance'])
    uw = np.asarray(uniform['analysis_weight'], dtype=np.float64)
    inside_u = ud <= epsilon
    t = uy == LABEL_T
    f = uy == LABEL_F
    u = uy == LABEL_U
    uniform_tpr = _weighted_ratio(inside_u, t, uw)
    uniform_fpr = _weighted_ratio(inside_u, f, uw)
    tp = float(uw[inside_u & t].sum())
    fp = float(uw[inside_u & f].sum())
    direct_precision = tp / (tp + fp) if tp + fp else 1.0
    tf_weight = float(uw[t | f].sum())
    prevalence_t = float(uw[t].sum() / tf_weight)
    reconstructed = (
        prevalence_t
        * uniform_tpr
        / (prevalence_t * uniform_tpr + (1.0 - prevalence_t) * uniform_fpr)
        if prevalence_t * uniform_tpr + (1.0 - prevalence_t) * uniform_fpr > 0
        else 1.0
    )
    total_weight = float(uw.sum())
    prevalence = {
        'T': float(uw[t].sum() / total_weight),
        'F': float(uw[f].sum() / total_weight),
        'U': float(uw[u].sum() / total_weight),
    }
    ignored_error = np.asarray(uniform['task_error'])[u]
    ignored_inside = inside_u[u]
    ignored_summary = {
        'ball_occupancy': _weighted_ratio(inside_u, u, uw),
        'inside_task_error_quantiles': _quantiles(
            ignored_error[ignored_inside]
        ),
        'outside_task_error_quantiles': _quantiles(
            ignored_error[~ignored_inside]
        ),
    }
    strata = np.asarray(stratified.get('negative_stratum', []))
    per_stratum = {}
    if len(strata):
        for index, name in enumerate(('boundary_outside', 'medium', 'far')):
            denominator = sy == LABEL_F
            denominator &= strata == index
            per_stratum[name] = _weighted_ratio(inside_s, denominator, sw)
    same = np.asarray(uniform['anchor_episode']) == np.asarray(
        uniform['goal_episode']
    )
    same_cross = {}
    for name, mask in (('same_trajectory', same), ('cross_trajectory', ~same)):
        same_cross[name] = {
            'tpr': _weighted_ratio(inside_u, t & mask, uw),
            'fpr': _weighted_ratio(inside_u, f & mask, uw),
            'count': int(mask.sum()),
        }
    return {
        'epsilon': float(epsilon),
        'macro_tpr': float(np.mean(group_tpr)),
        'macro_fpr': float(np.mean(group_fpr)),
        'pair_weighted_stratified_tpr': _weighted_ratio(
            inside_s, sy == LABEL_T, sw
        ),
        'pair_weighted_stratified_fpr': _weighted_ratio(
            inside_s, sy == LABEL_F, sw
        ),
        'uniform_tpr': uniform_tpr,
        'uniform_fpr': uniform_fpr,
        'population_precision_direct': direct_precision,
        'population_precision_reconstructed': reconstructed,
        'precision_reconstruction_abs_error': abs(
            direct_precision - reconstructed
        ),
        'uniform_prevalence': prevalence,
        'ignored_band': ignored_summary,
        'negative_fpr_by_stratum': per_stratum,
        'trajectory_strata': same_cross,
        '_group_tpr': np.asarray(group_tpr, dtype=np.float64),
        '_group_fpr': np.asarray(group_fpr, dtype=np.float64),
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values)
    if not len(values):
        return {'q05': None, 'q50': None, 'q95': None}
    q = np.quantile(values, [0.05, 0.5, 0.95])
    return {'q05': float(q[0]), 'q50': float(q[1]), 'q95': float(q[2])}


def bootstrap_macro_ci(
    group_values: np.ndarray, replicates: int, seed: int
) -> dict[str, float]:
    values = np.asarray(group_values, dtype=np.float64)
    if not len(values):
        return {'low': float('nan'), 'high': float('nan')}
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    chunk = 256
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        sampled = rng.integers(0, len(values), size=(count, len(values)))
        distribution[start : start + count] = values[sampled].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {'low': float(low), 'high': float(high)}


def attach_bootstrap_cis(metrics: dict, replicates: int, seed: int) -> dict:
    result = dict(metrics)
    tpr = result.pop('_group_tpr')
    fpr = result.pop('_group_fpr')
    result['group_clustered_95ci'] = {
        'macro_tpr': bootstrap_macro_ci(tpr, replicates, seed),
        'macro_fpr': bootstrap_macro_ci(fpr, replicates, seed + 1),
        'replicates': int(replicates),
        'seed': int(seed),
    }
    return result


def bootstrap_uniform_cis(
    uniform: dict[str, np.ndarray],
    epsilon: float,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Clustered CIs for pair-population and ignored-band estimands."""
    labels = np.asarray(uniform['label'])
    distance = np.asarray(uniform['latent_distance'])
    weights = np.asarray(uniform['analysis_weight'], dtype=np.float64)
    groups, inverse = np.unique(
        np.asarray(uniform['anchor_group']), return_inverse=True
    )
    inside = distance <= epsilon
    columns = []
    for mask in (
        labels == LABEL_T,
        labels == LABEL_F,
        labels == LABEL_U,
        inside & (labels == LABEL_T),
        inside & (labels == LABEL_F),
        inside & (labels == LABEL_U),
    ):
        columns.append(
            np.bincount(inverse, weights=weights * mask, minlength=len(groups))
        )
    components = np.stack(columns, axis=1)
    rng = np.random.default_rng(seed)
    names = (
        'uniform_tpr',
        'uniform_fpr',
        'population_precision_direct',
        'u_prevalence',
        'ignored_band_occupancy',
    )
    distributions = {name: np.empty(replicates) for name in names}
    chunk = 128
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        sampled = rng.integers(0, len(groups), size=(count, len(groups)))
        totals = components[sampled].sum(axis=1)
        t, f, u, hit_t, hit_f, hit_u = totals.T
        values = (
            np.divide(hit_t, t, out=np.zeros_like(t), where=t > 0),
            np.divide(hit_f, f, out=np.zeros_like(f), where=f > 0),
            np.divide(
                hit_t,
                hit_t + hit_f,
                out=np.ones_like(hit_t),
                where=(hit_t + hit_f) > 0,
            ),
            np.divide(
                u, t + f + u, out=np.zeros_like(u), where=(t + f + u) > 0
            ),
            np.divide(hit_u, u, out=np.zeros_like(u), where=u > 0),
        )
        for name, value in zip(names, values):
            distributions[name][start : start + count] = value
    return {
        name: {
            'low': float(np.quantile(values, 0.025)),
            'high': float(np.quantile(values, 0.975)),
        }
        for name, values in distributions.items()
    }


def bootstrap_epsilon_stability(
    stratified: dict[str, np.ndarray],
    *,
    min_positive_recall: float,
    max_negative_fpr: float,
    replicates: int,
    seed: int,
    histogram_bins: int = 512,
) -> tuple[dict, np.ndarray]:
    """Group bootstrap of epsilon using pre-aggregated distance histograms.

    This stability analysis is histogram-approximated and never replaces the
    exact deterministic threshold returned by :func:`select_threshold`.
    """
    labels = np.asarray(stratified['label'])
    distance = np.asarray(stratified['latent_distance'], dtype=np.float32)
    weights = np.asarray(stratified['analysis_weight'], dtype=np.float64)
    groups, inverse = np.unique(
        np.asarray(stratified['anchor_group']), return_inverse=True
    )
    keep = (labels == LABEL_T) | (labels == LABEL_F)
    quantiles = np.linspace(0.0, 1.0, histogram_bins)
    grid = np.unique(np.quantile(distance[keep], quantiles)).astype(np.float32)
    bins = np.searchsorted(grid, distance, side='left')
    n_bins = len(grid)
    per_group = []
    for target in (LABEL_T, LABEL_F):
        selected = labels == target
        flat = inverse[selected] * n_bins + bins[selected]
        histogram = np.bincount(
            flat,
            weights=weights[selected],
            minlength=len(groups) * n_bins,
        ).reshape(len(groups), n_bins)
        totals = histogram.sum(axis=1, keepdims=True)
        histogram = np.divide(
            histogram,
            totals,
            out=np.zeros_like(histogram),
            where=totals > 0,
        )
        per_group.append(histogram)
    rng = np.random.default_rng(seed)
    epsilon = np.full(replicates, np.nan, dtype=np.float64)
    chunk = 32
    probabilities = np.full(len(groups), 1.0 / len(groups))
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        multiplicity = rng.multinomial(
            len(groups), probabilities, size=count
        ).astype(np.float64)
        tpr = np.cumsum(multiplicity @ per_group[0], axis=1) / len(groups)
        fpr = np.cumsum(multiplicity @ per_group[1], axis=1) / len(groups)
        for row in range(count):
            feasible = fpr[row] <= max_negative_fpr
            if not feasible.any():
                continue
            best_tpr = tpr[row, feasible].max()
            if best_tpr < min_positive_recall:
                continue
            winner = np.flatnonzero(feasible & (tpr[row] == best_tpr))[0]
            epsilon[start + row] = float(grid[winner])
    finite = epsilon[np.isfinite(epsilon)]
    summary = {
        'replicates': int(replicates),
        'seed': int(seed),
        'histogram_bins_requested': int(histogram_bins),
        'histogram_bins_realized': len(grid),
        'feasible_fraction': float(len(finite) / replicates),
        'q025': float(np.quantile(finite, 0.025)) if len(finite) else None,
        'q50': float(np.quantile(finite, 0.5)) if len(finite) else None,
        'q975': float(np.quantile(finite, 0.975)) if len(finite) else None,
        'role': 'stability only; deterministic exact epsilon remains primary',
    }
    return summary, epsilon


def enforce_validation(
    metrics: dict,
    *,
    min_positive_recall: float,
    max_negative_fpr: float,
    min_population_precision: float | None,
) -> None:
    failures = []
    if metrics['macro_tpr'] < min_positive_recall:
        failures.append('macro_tpr')
    if metrics['macro_fpr'] > max_negative_fpr:
        failures.append('macro_fpr')
    if (
        min_population_precision is not None
        and metrics['population_precision_direct'] < min_population_precision
    ):
        failures.append('population_precision_direct')
    if failures:
        raise CalibrationOutcome(
            'THRESHOLD_CALIBRATION_FAILED_VALIDATION',
            {'failed_constraints': failures, 'metrics': metrics},
        )
