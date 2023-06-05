from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


# TODO: Remove the `safe_*` prefix from the properties, given that the properties shadow the attribute variables (test this).
@dataclass
class Batch:
    """
    A batch of observations.
    ---
    The first dimension of each tensor represents the different observations of the batch.
    A single-observation batch is represented by the first dimension of each tensor being 1.
    ---
    Each of the tensors of the batch can be `None`; this allows the user to create batches with only the tensors they need.
    Each of the tensors of the batch can be safely retrieved by using the `safe_*` properties, without the need to check for `None` values.
    """

    device: Union[torch.device, str]
    """The device on which the tensors of the batch are stored."""

    states: Optional[torch.Tensor] = None
    """The states of the batch."""

    next_states: Optional[torch.Tensor] = None
    """The next states of the batch."""

    markers: Optional[torch.Tensor] = None
    """The markers of the batch common across all morphologies."""

    next_markers: Optional[torch.Tensor] = None
    """The next markers of the batch common across all morphologies."""

    actions: Optional[torch.Tensor] = None
    """The actions of the batch."""

    rewards: Optional[torch.Tensor] = None
    """The rewards of the batch."""

    terminateds: Optional[torch.Tensor] = None
    """The termination flags of the batch."""

    truncateds: Optional[torch.Tensor] = None
    """The truncation flags of the batch."""

    morphos: Optional[torch.Tensor] = None
    """The morphology parameters of the batch."""

    def __post_init__(self):
        assert (
            isinstance(self.states, Optional[torch.Tensor])
            and isinstance(self.next_states, Optional[torch.Tensor])
            and isinstance(self.markers, Optional[torch.Tensor])
            and isinstance(self.next_markers, Optional[torch.Tensor])
            and isinstance(self.actions, Optional[torch.Tensor])
            and isinstance(self.rewards, Optional[torch.Tensor])
            and isinstance(self.terminateds, Optional[torch.Tensor])
            and isinstance(self.truncateds, Optional[torch.Tensor])
            and isinstance(self.morphos, Optional[torch.Tensor])
        ), "The tensors of a batch must be of type torch.Tensor or None."

        # Get the size of the batch
        self._len = 0
        if self.states is not None:
            self._len = self.states.shape[0]
        elif self.next_states is not None:
            self._len = self.next_states.shape[0]
        elif self.markers is not None:
            self._len = self.markers.shape[0]
        elif self.next_markers is not None:
            self._len = self.next_markers.shape[0]
        elif self.actions is not None:
            self._len = self.actions.shape[0]
        elif self.rewards is not None:
            self._len = self.rewards.shape[0]
        elif self.terminateds is not None:
            self._len = self.terminateds.shape[0]
        elif self.truncateds is not None:
            self._len = self.truncateds.shape[0]
        elif self.morphos is not None:
            self._len = self.morphos.shape[0]

        if self._len == 0:
            raise ValueError("At least one tensor of the batch must be not None.")

        err_msg = "The first dimension of each tensor of a batch must be the same."
        assert self.states is None or self.states.shape[0] == self._len, err_msg
        assert (
            self.next_states is None or self.next_states.shape[0] == self._len
        ), err_msg
        assert self.markers is None or self.markers.shape[0] == self._len, err_msg
        assert (
            self.next_markers is None or self.next_markers.shape[0] == self._len
        ), err_msg
        assert self.actions is None or self.actions.shape[0] == self._len, err_msg
        assert self.rewards is None or self.rewards.shape[0] == self._len, err_msg
        assert (
            self.terminateds is None or self.terminateds.shape[0] == self._len
        ), err_msg
        assert self.truncateds is None or self.truncateds.shape[0] == self._len, err_msg
        assert self.morphos is None or self.morphos.shape[0] == self._len, err_msg

        self.states = self.states.to(self.device) if self.states is not None else None
        self.next_states = (
            self.next_states.to(self.device) if self.next_states is not None else None
        )
        self.markers = (
            self.markers.to(self.device) if self.markers is not None else None
        )
        self.next_markers = (
            self.next_markers.to(self.device) if self.next_markers is not None else None
        )
        self.actions = (
            self.actions.to(self.device) if self.actions is not None else None
        )
        self.rewards = (
            self.rewards.to(self.device) if self.rewards is not None else None
        )
        self.terminateds = (
            self.terminateds.to(self.device) if self.terminateds is not None else None
        )
        self.truncateds = (
            self.truncateds.to(self.device) if self.truncateds is not None else None
        )
        self.morphos = (
            self.morphos.to(self.device) if self.morphos is not None else None
        )
        self.features = (
            torch.cat((self.states, self.morphos), dim=1).to(self.device)
            if self.states is not None and self.morphos is not None
            else None
        )
        self.next_features = (
            torch.cat((self.next_states, self.morphos), dim=1).to(self.device)
            if self.next_states is not None and self.morphos is not None
            else None
        )

    def __len__(self) -> int:
        return self._len

    @property
    def safe_states(self) -> torch.Tensor:
        assert self.states is not None, "The states of the batch must be not None."
        return self.states

    @property
    def safe_next_states(self) -> torch.Tensor:
        assert (
            self.next_states is not None
        ), "The next states of the batch must be not None."
        return self.next_states

    @property
    def safe_markers(self) -> torch.Tensor:
        assert self.markers is not None, "The markers of the batch must be not None."
        return self.markers

    @property
    def safe_next_markers(self) -> torch.Tensor:
        assert (
            self.next_markers is not None
        ), "The next markers of the batch must be not None."
        return self.next_markers

    @property
    def safe_actions(self) -> torch.Tensor:
        assert self.actions is not None, "The actions of the batch must be not None."
        return self.actions

    @property
    def safe_rewards(self) -> torch.Tensor:
        assert self.rewards is not None, "The rewards of the batch must be not None."
        return self.rewards

    @property
    def safe_terminateds(self) -> torch.Tensor:
        assert (
            self.terminateds is not None
        ), "The terminateds of the batch must be not None."
        return self.terminateds

    @property
    def safe_truncateds(self) -> torch.Tensor:
        assert (
            self.truncateds is not None
        ), "The truncateds of the batch must be not None."
        return self.truncateds

    @property
    def safe_morphos(self) -> torch.Tensor:
        assert self.morphos is not None, "The morphos of the batch must be not None."
        return self.morphos

    @property
    def safe_features(self) -> torch.Tensor:
        """
        The concatenation of the states and the morphology parameters.
        """

        assert self.features is not None, "The features of the batch must be not None."
        return self.features

    @property
    def safe_next_features(self) -> torch.Tensor:
        """
        The concatenation of the next states and the morphology parameters.
        """

        assert (
            self.next_features is not None
        ), "The next features of the batch must be not None."
        return self.next_features

    def safe_features_with(self, morpho: torch.Tensor) -> torch.Tensor:
        """
        The concatenation of the states and the morphology parameters.

        Parameters
        ----------
        morpho -> the morphology parameters to concatenate to the states.
        """

        assert self.states is not None, "The states of the batch must be not None."

        rep_shape = [self._len] + ([1] * len(morpho.shape))
        rep_morpho = morpho.unsqueeze(0).repeat(*rep_shape)
        return torch.cat((self.states, rep_morpho), dim=1)

    def safe_next_features_with(self, morpho: torch.Tensor) -> torch.Tensor:
        """
        The concatenation of the next states and the morphology parameters.

        Parameters
        ----------
        morpho -> the morphology parameters to concatenate to the next states.
        """

        assert (
            self.next_states is not None
        ), "The next states of the batch must be not None."

        rep_shape = [self._len] + ([1] * len(morpho.shape))
        rep_morpho = morpho.unsqueeze(0).repeat(*rep_shape)
        return torch.cat((self.next_states, rep_morpho), dim=1)

    def to_list(self) -> list[Batch]:
        """
        Returns the batch as a list of single-observation batches, where each element of the list corresponds to each element of the first dimension of each tensor of the `Batch`.
        """

        return [
            Batch(
                states=self.states[i, :].unsqueeze(0)
                if self.states is not None
                else None,
                next_states=self.next_states[i, :].unsqueeze(0)
                if self.next_states is not None
                else None,
                markers=self.markers[i, :].unsqueeze(0)
                if self.markers is not None
                else None,
                next_markers=self.next_markers[i, :].unsqueeze(0)
                if self.next_markers is not None
                else None,
                actions=self.actions[i, :].unsqueeze(0)
                if self.actions is not None
                else None,
                rewards=self.rewards[i, :].unsqueeze(0)
                if self.rewards is not None
                else None,
                terminateds=self.terminateds[i, :].unsqueeze(0)
                if self.terminateds is not None
                else None,
                truncateds=self.truncateds[i, :].unsqueeze(0)
                if self.truncateds is not None
                else None,
                morphos=self.morphos[i, :].unsqueeze(0)
                if self.morphos is not None
                else None,
                device=self.device,
            )
            for i in range(self._len)
        ]

    @classmethod
    def merge(cls, batch_list: list[Batch], device: Union[torch.device, str]) -> Batch:
        """
        Merges a list of batches into a single batch.
        The first dimension of each tensor represents the number of observations in the batch.

        Parameters
        ----------
        `batch_list` -> the list of `Batch` objects to merge.
        `device` -> the device on which the tensors of the batch are stored.

        Returns
        -------
        The merged `Batch` object.
        """

        assert len(batch_list) > 0, "The list of batches must be not empty."

        return cls(
            states=torch.cat(
                [batch.states for batch in batch_list if batch.states is not None],
                dim=0,
            ),
            next_states=torch.cat(
                [
                    batch.next_states
                    for batch in batch_list
                    if batch.next_states is not None
                ],
                dim=0,
            ),
            markers=torch.cat(
                [batch.markers for batch in batch_list if batch.markers is not None],
                dim=0,
            ),
            next_markers=torch.cat(
                [
                    batch.next_markers
                    for batch in batch_list
                    if batch.next_markers is not None
                ],
                dim=0,
            ),
            actions=torch.cat(
                [batch.actions for batch in batch_list if batch.actions is not None],
                dim=0,
            ),
            rewards=torch.cat(
                [batch.rewards for batch in batch_list if batch.rewards is not None],
                dim=0,
            ),
            terminateds=torch.cat(
                [
                    batch.terminateds
                    for batch in batch_list
                    if batch.terminateds is not None
                ],
                dim=0,
            ),
            truncateds=torch.cat(
                [
                    batch.truncateds
                    for batch in batch_list
                    if batch.truncateds is not None
                ],
                dim=0,
            ),
            morphos=torch.cat(
                [batch.morphos for batch in batch_list if batch.morphos is not None],
                dim=0,
            ),
            device=device,
        )

    @classmethod
    def from_numpy(
        cls,
        states: Optional[np.ndarray],
        next_states: Optional[np.ndarray],
        markers: Optional[np.ndarray],
        next_markers: Optional[np.ndarray],
        actions: Optional[np.ndarray],
        rewards: Optional[np.ndarray],
        terminateds: Optional[np.ndarray],
        truncateds: Optional[np.ndarray],
        morphos: Optional[np.ndarray],
        device: Union[torch.device, str],
    ) -> Batch:
        """
        Creates a `Batch` object from a batch of observations represented as `np.ndarray` objects.

        Parameters
        ----------
        `device` -> the device to which the tensors of the batch will be sent.
        `states` -> the states of the batch.
        `next_states` -> the next states of the batch.
        `markers` -> the markers of the batch.
        `next_markers` -> the next markers of the batch.
        `actions` -> the actions of the batch.
        `rewards` -> the rewards of the batch.
        `terminateds` -> the termination flags of the batch.
        `truncateds` -> the truncation flags of the batch.
        `morphos` -> the morphology parameters of the batch.

        Returns
        -------
        The `Batch` object.
        """

        states_t = (
            None if states is None else torch.from_numpy(states).float().to(device)
        )
        next_states_t = (
            None
            if next_states is None
            else torch.from_numpy(next_states).float().to(device)
        )
        markers_t = (
            None if markers is None else torch.from_numpy(markers).float().to(device)
        )
        next_markers_t = (
            None
            if next_markers is None
            else torch.from_numpy(next_markers).float().to(device)
        )
        actions_t = (
            None if actions is None else torch.from_numpy(actions).float().to(device)
        )
        rewards_t = (
            None
            if rewards is None
            else torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        )
        terminateds_t = (
            None
            if terminateds is None
            else torch.from_numpy(terminateds).float().unsqueeze(1).to(device)
        )
        truncateds_t = (
            None
            if truncateds is None
            else torch.from_numpy(truncateds).float().unsqueeze(1).to(device)
        )
        morphos_t = (
            None if morphos is None else torch.from_numpy(morphos).float().to(device)
        )

        return cls(
            states=states_t,
            next_states=next_states_t,
            markers=markers_t,
            next_markers=next_markers_t,
            actions=actions_t,
            rewards=rewards_t,
            terminateds=terminateds_t,
            truncateds=truncateds_t,
            morphos=morphos_t,
            device=device,
        )
