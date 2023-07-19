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
        optimized_demonstrator: bool = True,
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
        self._batch_demonstrator = None

    @property
    def demonstrator(self) -> torch.Tensor:
        """The demonstrator morphology parameters."""

        assert (
            self._demonstrator is not None
        ), "Must call `adapt` before accessing the demonstrator morphology."
        return self._demonstrator

    @property
    def batch_demonstrator(self) -> torch.Tensor:
        """
        The demonstrator morphology parameters as a batch.
        The first dimension is of the size of the `batch_size` parameter passed to the last `adapt` call.
        """

        assert (
            self._batch_demonstrator is not None
        ), "Must call `adapt` before accessing the demonstrator morphology."
        return self._batch_demonstrator

    def train(self, *_) -> Tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    @torch.no_grad()
    def _loss(
        self,
        batch: tuple,
        morphos: torch.Tensor,
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ) -> np.ndarray:
        q_function.eval()
        policy.eval()

        (feats_batch, action_batch, reward_batch, next_feats_batch, *_) = batch
        batch_shape = feats_batch.shape[0]

        if len(morphos.shape) == 1:
            morphos = morphos.unsqueeze(0)
        n_morphos = morphos.shape[0]

        losses = np.zeros(n_morphos)
        for i in range(n_morphos):
            morpho = morphos[i].unsqueeze(0)

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

            losses[i] = torch.mean(
                torch.abs(q_vals - gamma * next_q_vals - reward_batch)
            ).item()

        return losses

    def get_demonstrators_for(
        self,
        batch: tuple,
        prev_morphos: list[np.ndarray],
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ) -> torch.Tensor:
        """
        Returns the best demonstrator for each observation in the given batch, out of all previous morphologies.

        Parameters
        ----------
        `batch` -> the batch of observations.
        `prev_morphos` -> the previous morphologies.
        `q_function` -> the Q-function to use for the loss.
        `policy` -> the policy to use for the loss.
        `gamma` -> the discount factor to use for the loss.
        """

        assert (
            len(prev_morphos) > 0
        ), "Must have at least one previous morphology to search for the best demonstrator."

        batch_size = batch[0].shape[0]
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

        prev_morphos_t = [
            torch.tensor(m, dtype=torch.float32, device=self._device)
            for m in prev_morphos
        ]

        best_demonstrators = torch.zeros(batch_size, *(prev_morphos_t[0].shape)).to(
            self._device
        )

        for i in range(batch_size):
            best_loss = float("inf")
            for prev_morpho in prev_morphos_t:
                prev_morpho = prev_morpho.to(self._device)
                obs = (
                    batch[0][i].unsqueeze(0),
                    batch[1][i].unsqueeze(0),
                    batch[2][i].unsqueeze(0),
                    batch[3][i].unsqueeze(0),
                    None,
                )
                loss = self._loss(obs, prev_morpho, q_function, policy, gamma)[0]
                if loss < best_loss:
                    best_loss = loss
                    best_demonstrators[i] = prev_morpho

        return best_demonstrators.detach()

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
            loss = self._loss(batch, prev_morpho, q_function, policy, gamma)[0]
            if loss < best_loss:
                best_loss = loss
                best_demonstrator = prev_morpho

        return best_demonstrator

    def _optimize_best_demonstrator(
        self,
        batch: tuple,
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ) -> torch.Tensor:
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=10,
            dimensions=self._bounds.shape[0],
            options=options,
            bounds=(self._bounds[:, 0].cpu().numpy(), self._bounds[:, 1].cpu().numpy()),
            ftol=1e-7,
            ftol_iter=30,
        )

        @torch.no_grad()
        def fn(morphos: np.ndarray):
            morpho_t = torch.tensor(morphos, dtype=torch.float32, device=self._device)
            losses = self._loss(batch, morpho_t, q_function, policy, gamma)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            return losses

        _, pos = optimizer.optimize(fn, iters=100)
        optimized_demonstrator = torch.tensor(
            pos, dtype=torch.float32, device=self._device
        )

        return optimized_demonstrator

    def adapt(
        self,
        batch: tuple,
        batch_size: int,
        prev_morphos: list[np.ndarray],
        q_function: EnsembleQNetwork,
        policy: GaussianPolicy,
        gamma: float,
    ) -> None:
        """
        Make an adaptation step, where the rewarder adapts its internal state when the morphology changes.

        Parameters
        ----------
        `batch` -> a batch of data collected using the new morphology.
        `batch_size` -> the size that will be used as the first dimension of the `batch_demonstrator` tensor.
        `prev_morphos` -> the previous morphologies.
        `q_function` -> the Q-function used by the agent to compute the Q-values.
        - It must be take as input the concatenation of the state and the morphology parameters, as well as the action, and return the Q-value.
        `policy` -> the policy used by the agent to compute the actions.
        - It must be take as input the concatenation of the state and the morphology parameters, and return the action.
        `gamma` -> the discount factor in the Q-value formula.
        """

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
            self._demonstrator = self._optimize_best_demonstrator(
                batch, q_function, policy, gamma
            ).detach()
        else:
            self._demonstrator = self._search_best_demonstrator(
                batch, prev_morphos, q_function, policy, gamma
            ).detach()

        batch_demonstrator = self._demonstrator
        if len(batch_demonstrator.shape) == 1:
            batch_demonstrator = batch_demonstrator.unsqueeze(0)
        if batch_demonstrator.shape[0] != batch_size:
            new_shape = [batch_size] + ([1] * len(batch_demonstrator.shape[1:]))
            batch_demonstrator = batch_demonstrator.repeat(*new_shape)
        self._batch_demonstrator = batch_demonstrator

    def _compute_rewards_impl(self, batch: tuple, demos: tuple) -> torch.Tensor:
        if self._demonstrator is None:
            raise RuntimeError(
                "Must call adapt() before computing rewards for a batch."
            )

        action_batch = batch[1]
        (_, action_demos, *_) = demos

        diff = action_batch - action_demos
        summed_diff = diff.sum(dim=1)
        rewards = -0.5 * torch.square(summed_diff)
        return rewards.unsqueeze(1)

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {
            "demonstrator": self._demonstrator,
            "batch_demonstrator": self._batch_demonstrator,
            "bounds": self._bounds,
        }

    def _load_impl(self, model: Dict[str, Any]):
        try:
            self._demonstrator = model["demonstrator"]
            self._batch_demonstrator = model["batch_demonstrator"]
            self._bounds = model["bounds"]
        except KeyError:
            raise ValueError("Invalid MBC model")
