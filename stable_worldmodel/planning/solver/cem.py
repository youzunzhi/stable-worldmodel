"""Cross Entropy Method solver for model-based planning."""

import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .callbacks import Callback
from .solver import Costable
from .utils import prepare_init_action


class CEMSolver:
    """Cross Entropy Method solver for action optimization.

    Args:
        cost: Cost object to plan against (a Costable, e.g. a ShootingCostEvaluator).
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for the action distribution.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        cost: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        callbacks: list[Callback] | None = None,
        candidate_noise: Callable[..., torch.Tensor] | None = None,
        candidate_transform: Callable[[torch.Tensor], torch.Tensor]
        | None = None,
        iteration_observer: Callable[..., None] | None = None,
        verified_hit_key: str | None = None,
        return_best_evaluated: bool = False,
        log_timing: bool = True,
    ) -> None:
        self.cost = cost
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.callbacks = list(callbacks) if callbacks else []
        self.candidate_noise = candidate_noise
        self.candidate_transform = candidate_transform
        self.iteration_observer = iteration_observer
        self.verified_hit_key = verified_hit_key
        self.return_best_evaluated = bool(return_best_evaluated)
        if (
            self.verified_hit_key is not None
            and self.iteration_observer is None
        ):
            raise ValueError(
                'verified-hit selection requires an iteration observer'
            )
        self.log_timing = bool(log_timing)
        try:
            self._dtype = next(cost.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f'Action space is discrete, got {type(action_space)}. CEMSolver may not work as expected.'
            )

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the action distribution parameters (mean and variance)."""
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim], dtype=self.dtype
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
            if actions is None
            else actions
        )

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim], dtype=self.dtype
            )
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using Cross Entropy Method."""
        start_time = time.time()
        outputs = {
            'costs': [],
            'mean': [],  # History of means
            'var': [],  # History of vars
        }
        returned_actions = []
        returned_verified_hits = []
        return_modes = []

        # Batch size is taken from info_dict so callers can solve for a subset of envs
        total_envs = len(next(iter(info_dict.values())))

        # -- warm-start from actor if model is Actionable, else zero-pad
        init_action = prepare_init_action(
            self.cost,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )

        # -- initialize the action distribution globally
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        for cb in self.callbacks:
            cb.reset()
        if self.iteration_observer is not None:
            reset = getattr(self.iteration_observer, 'reset', None)
            if callable(reset):
                reset()

        # --- Iterate over batches ---
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            # Slice Distribution Parameters for current batch
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand Info Dict for current batch
            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = (
                        self.dtype if v_batch.is_floating_point() else None
                    )
                    v_batch = (
                        v_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(
                            current_bs,
                            self.num_samples,
                            *v_batch.shape[1:],
                        )
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(
                        v_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            # Optimization Loop
            final_batch_cost = None
            best_cost = torch.full(
                (current_bs,),
                float('inf'),
                device=self.device,
                dtype=self.dtype,
            )
            best_actions = torch.zeros_like(batch_mean)
            archive_cost = torch.full_like(best_cost, float('inf'))
            archive_actions = torch.zeros_like(batch_mean)
            archive_found = torch.zeros(
                current_bs, device=self.device, dtype=torch.bool
            )

            for cb in self.callbacks:
                cb.start_batch()
            if self.iteration_observer is not None:
                start_batch = getattr(
                    self.iteration_observer, 'start_batch', None
                )
                if callable(start_batch):
                    start_batch(start_idx=start_idx, end_idx=end_idx)

            for step in range(self.n_steps):
                # Sample action sequences: (Batch, Num_Samples, Horizon, Dim)
                noise_shape = (
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                )
                if self.candidate_noise is None:
                    candidates = torch.randn(
                        *noise_shape,
                        generator=self.torch_gen,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    candidates = self.candidate_noise(
                        step=step,
                        batch_start=start_idx,
                        shape=noise_shape,
                        device=torch.device(self.device),
                        dtype=self.dtype,
                    )
                    if not torch.is_tensor(candidates):
                        raise TypeError(
                            'candidate_noise must return a torch.Tensor'
                        )
                    if tuple(candidates.shape) != noise_shape:
                        raise ValueError(
                            'candidate_noise returned shape '
                            f'{tuple(candidates.shape)}, expected {noise_shape}'
                        )
                    expected_device = torch.device(self.device)
                    device_mismatch = (
                        candidates.device.type != expected_device.type
                        or (
                            expected_device.index is not None
                            and candidates.device.index
                            != expected_device.index
                        )
                    )
                    if device_mismatch:
                        raise ValueError(
                            'candidate_noise returned a tensor on '
                            f'{candidates.device}, expected {self.device}'
                        )
                    if candidates.dtype != self.dtype:
                        raise ValueError(
                            'candidate_noise returned dtype '
                            f'{candidates.dtype}, expected {self.dtype}'
                        )

                # Scale and shift: (Batch, N, H, D) * (Batch, 1, H, D) + (Batch, 1, H, D)
                candidates = candidates * batch_var.unsqueeze(
                    1
                ) + batch_mean.unsqueeze(1)

                # Force the first sample to be the current mean
                candidates[:, 0] = batch_mean
                if self.candidate_transform is not None:
                    transformed = self.candidate_transform(candidates)
                    if not torch.is_tensor(transformed):
                        raise TypeError(
                            'candidate_transform must return a torch.Tensor'
                        )
                    if transformed.shape != candidates.shape:
                        raise ValueError(
                            'candidate_transform changed candidate shape from '
                            f'{tuple(candidates.shape)} to '
                            f'{tuple(transformed.shape)}'
                        )
                    candidates = transformed

                # Evaluate candidates
                if self.iteration_observer is None:
                    costs = self.cost.get_cost(expanded_infos, candidates)
                else:
                    evaluate = getattr(
                        self.cost, 'get_cost_and_diagnostics', None
                    )
                    if not callable(evaluate):
                        raise TypeError(
                            'iteration_observer requires cost to expose '
                            'get_cost_and_diagnostics'
                        )
                    costs, diagnostics = evaluate(expanded_infos, candidates)
                    self.iteration_observer(
                        step=step,
                        batch_start=start_idx,
                        diagnostics=diagnostics,
                    )

                current_vals, current_inds = costs.min(dim=1)
                current_actions = candidates[
                    torch.arange(current_bs, device=self.device), current_inds
                ]
                replace_best = current_vals < best_cost
                best_cost = torch.where(replace_best, current_vals, best_cost)
                best_actions = torch.where(
                    replace_best[:, None, None],
                    current_actions,
                    best_actions,
                )

                assert isinstance(costs, torch.Tensor), (
                    f'Expected cost to be a torch.Tensor, got {type(costs)}'
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples
                ), (
                    f'Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                )

                # Select Top-K
                # topk_vals: (Batch, K), topk_inds: (Batch, K)
                hit_bits = None
                if self.verified_hit_key is not None:
                    hit_bits = diagnostics.get(self.verified_hit_key)
                    if (
                        not torch.is_tensor(hit_bits)
                        or hit_bits.dtype != torch.bool
                        or hit_bits.shape != costs.shape
                    ):
                        raise ValueError(
                            f'diagnostic {self.verified_hit_key!r} must be a '
                            'boolean tensor matching costs'
                        )
                    hit_costs = costs.masked_fill(~hit_bits, float('inf'))
                    iteration_hit_cost, iteration_hit_ind = hit_costs.min(
                        dim=1
                    )
                    iteration_has_hit = torch.isfinite(iteration_hit_cost)
                    iteration_hit_actions = candidates[
                        torch.arange(current_bs, device=self.device),
                        iteration_hit_ind,
                    ]
                    replace_archive = iteration_has_hit & (
                        iteration_hit_cost < archive_cost
                    )
                    archive_cost = torch.where(
                        replace_archive, iteration_hit_cost, archive_cost
                    )
                    archive_actions = torch.where(
                        replace_archive[:, None, None],
                        iteration_hit_actions,
                        archive_actions,
                    )
                    archive_found |= iteration_has_hit

                    # Stable two-pass ordering implements the exact tuple
                    # ``(not hit, cost, candidate index)`` without relying on
                    # an arbitrary large numeric penalty.
                    cost_order = torch.argsort(costs, dim=1, stable=True)
                    ordered_hits = hit_bits.gather(1, cost_order)
                    priority_order = torch.argsort(
                        (~ordered_hits).to(torch.int8), dim=1, stable=True
                    )
                    topk_inds = cost_order.gather(1, priority_order)[
                        :, : self.topk
                    ]
                    topk_vals = costs.gather(1, topk_inds)
                else:
                    topk_vals, topk_inds = torch.topk(
                        costs, k=self.topk, dim=1, largest=False
                    )

                after_selection = getattr(
                    self.iteration_observer, 'after_selection', None
                )
                if callable(after_selection):
                    after_selection(
                        step=step,
                        batch_start=start_idx,
                        diagnostics=diagnostics,
                        topk_inds=topk_inds,
                    )

                # Gather Top-K Candidates
                # We need to select the specific candidates corresponding to topk_inds
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )

                # Indexing: candidates[batch_idx, sample_idx]
                # Result shape: (Batch, K, Horizon, Dim)
                topk_candidates = candidates[batch_indices, topk_inds]

                # Update Mean and Variance based on Top-K
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                for cb in self.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )

                # Update final cost for logging
                # We average the cost of the top elites
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            # Write results back to global storage
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

            if self.verified_hit_key is not None:
                selected = torch.where(
                    archive_found[:, None, None],
                    archive_actions,
                    best_actions,
                )
                returned_actions.append(selected.detach().cpu())
                returned_verified_hits.extend(
                    archive_found.detach().cpu().tolist()
                )
                return_modes.extend(
                    [
                        'verified_hit_archive'
                        if bool(value)
                        else 'best_evaluated_candidate'
                        for value in archive_found.detach().cpu().tolist()
                    ]
                )
            elif self.return_best_evaluated:
                returned_actions.append(best_actions.detach().cpu())
                returned_verified_hits.extend([False] * current_bs)
                return_modes.extend(['best_evaluated_candidate'] * current_bs)

            # Store history/metadata
            outputs['costs'].extend(final_batch_cost)

        outputs['distribution_mean'] = mean.detach().cpu()
        outputs['actions'] = (
            torch.cat(returned_actions, dim=0)
            if returned_actions
            else mean.detach().cpu()
        )
        if returned_actions:
            outputs['returned_verified_hit'] = returned_verified_hits
            outputs['return_mode'] = return_modes
        outputs['mean'] = [mean.detach().cpu()]
        outputs['var'] = [var.detach().cpu()]

        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history
        if self.iteration_observer is not None:
            end_solve = getattr(self.iteration_observer, 'end_solve', None)
            if callable(end_solve):
                outputs['iteration_observer'] = end_solve()

        if self.log_timing:
            print(f'CEM solve time: {time.time() - start_time:.4f} seconds')
        return outputs
