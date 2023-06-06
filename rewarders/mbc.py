from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyswarms as ps
import torch

from common.models import EnsembleQNetwork, GaussianPolicy
from normalizers import Normalizer

from .rewarder import Rewarder


class MBC(Rewarder):
    """
    The morphological behavioral cloning rewarder.
    """

    def __init__(
        self,
        device: str,
        bounds: torch.Tensor,
        optimized_demonstrator: bool = False,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Initializes the MBC rewarder.

        Parameters
        ----------
        `device` -> the device to use for the torch tensors.
        `optimized_demonstrator` -> whether to optimize the demonstrator morphology (True) or to choose the best fitting existing previous morphology (False).
        `normalizer` -> the normalizer to use for the rewarder.
        """
        super().__init__(normalizer)

        self._optimized_demonstrator = optimized_demonstrator
        self._device = torch.device(device)
        self._bounds = bounds

        self._demonstrator = None

    @property
    def demonstrator(self) -> torch.Tensor:
        """
        The demonstrator morphology parameters.
        """

        assert (
            self._demonstrator is not None
        ), "Must call `co-adapt` before accessing the demonstrator morphology."
        return self._demonstrator

    def train(self, *_) -> Tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    @torch.no_grad()
    def _loss(
        self,
        batch: tuple,
        morpho: torch.Tensor,
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ):
        q_function.eval()
        policy.eval()

        (feats_batch, action_batch, reward_batch, next_feats_batch, *_) = batch

        if len(morpho.shape) == 1:
            morpho = morpho.unsqueeze(0)

        batch_shape = feats_batch.shape[0]
        if morpho.shape[0] != batch_shape:
            new_shape = [batch_shape] + ([1] * len(morpho.shape[1:]))
            morpho = morpho.repeat(*new_shape)

        morpho_size = morpho.shape[1]

        states_batch = feats_batch[:, :-morpho_size]
        next_states_batch = next_feats_batch[:, :-morpho_size]
        morpho_feats_batch = torch.cat([states_batch, morpho], dim=1)
        morpho_next_feats_batch = torch.cat([next_states_batch, morpho], dim=1)

        q_vals = q_function.min(morpho_feats_batch, action_batch)

        _, _, next_actions, _ = policy.sample(morpho_next_feats_batch)
        next_q_vals = q_function.min(morpho_next_feats_batch, next_actions)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return torch.mean(torch.pow(q_vals - gamma * next_q_vals - reward_batch, 2))

    def _search_best_demonstrator(
        self,
        batch: tuple,
        prev_morphos: list[np.ndarray],
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ) -> torch.Tensor:
        assert (
            len(prev_morphos) > 0
        ), "Must have at least one previous morphology to search for the best demonstrator."

        prev_morphos_t = [
            torch.tensor(m, dtype=torch.float32, device=self._device)
            for m in prev_morphos
        ]

        best_demonstrator = torch.zeros_like(prev_morphos_t[0])
        best_loss = float("inf")

        for prev_morpho in prev_morphos_t:
            prev_morpho = prev_morpho.to(self._device)
            loss = self._loss(batch, prev_morpho, q_function, policy, gamma)
            if loss < best_loss:
                best_loss = loss
                best_demonstrator = prev_morpho

        return best_demonstrator

    def _optimize_best_demonstrator(
        self,
    ) -> torch.Tensor:
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=250,
            dimensions=self._bounds.shape[0],
            options=options,
            bounds=(self._bounds[:, 0].cpu().numpy(), self._bounds[:, 1].cpu().numpy()),
            ftol=1e-7,
            ftol_iter=30,
        )
        _, pos = optimizer.optimize(self._loss, iters=250)
        optimized_demonstrator = torch.tensor(pos, device=self._device)

        return optimized_demonstrator

    def co_adapt(
        self,
        batch: tuple,
        prev_morphos: list[np.ndarray],
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ) -> None:
        """
        Make a co-adaptation step, where the rewarder adapts its internal state when the morphology changes.

        Parameters
        ----------
        `batch` -> a batch of data collected using the new morphology.
        `prev_morphos` -> the previous morphologies.
        `q_function` -> the Q-function used by the agent to compute the Q-values.
        - It must be take as input the concatenation of the state and the morphology parameters, as well as the action, and return the Q-value.
        `policy` -> the policy used by the agent to compute the actions.
        - It must be take as input the concatenation of the state and the morphology parameters, and return the action.
        `gamma` -> the discount factor in the Q-value formula.
        """

        assert (
            len(prev_morphos) > 0
        ), "Must have at least one previous morphology to co-adapt."

        feats_batch = torch.FloatTensor(batch[0]).to(self._device)
        action_batch = torch.FloatTensor(batch[1]).to(self._device)
        reward_batch = torch.FloatTensor(batch[2]).to(self._device).unsqueeze(1)
        next_feats_batch = torch.FloatTensor(batch[3]).to(self._device)

        batch = (
            feats_batch,
            action_batch,
            reward_batch,
            next_feats_batch,
            *batch[4:],
        )

        if self._optimized_demonstrator:
            self._demonstrator = self._optimize_best_demonstrator()
        else:
            self._demonstrator = self._search_best_demonstrator(
                batch, prev_morphos, q_function, policy, gamma
            )

    def _compute_rewards_impl(self, batch: tuple, demos: tuple) -> torch.Tensor:
        if self._demonstrator is None:
            raise RuntimeError(
                "Must call co-adapt() before computing rewards for a batch."
            )

        action_batch = torch.FloatTensor(batch[1]).to(self._device)
        (_, action_demos, *_) = demos

        return torch.mean(torch.pow(action_batch - action_demos, 2))

    def get_model_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load(self, model: Dict[str, Any]) -> None:
        raise NotImplementedError()
