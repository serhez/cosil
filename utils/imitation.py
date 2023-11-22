import glob
import os
from typing import List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from common.models import Discriminator, GaussianPolicy
from utils.co_adaptation import get_marker_info


def load_demos(config: DictConfig):
    demos, to_match, mean_reward = [], None, 0

    if os.path.isdir(config.method.expert_demos):
        for filepath in glob.iglob(
            f"{config.method.expert_demos}/expert_cmu_{config.method.subject_id}*.pt"
        ):
            episode = torch.load(filepath)
            episode_obs_np, to_match = get_marker_info(
                episode,
                config.method.expert_legs,
                config.method.expert_markers,
                pos_type=config.method.pos_type,
                vel_type=config.method.vel_type,
                torso_type=config.method.torso_type,
                head_type=config.method.head_type,
                head_wrt=config.method.head_wrt,
            )
            episode_obs = torch.from_numpy(episode_obs_np).float().to(config.device)
            demos.append(episode_obs)
    else:
        expert_obs = torch.load(config.method.expert_demos)
        # TODO: Consider using the mean reward without subtracting the penalty
        # if config.method.rm_action_penalty:
        mean_reward = np.mean(expert_obs["reward_run"])
        print(expert_obs.keys()) # TODO: Remove
        exit() # TODO: Remove
        expert_obs_np, to_match = get_marker_info(
            expert_obs,
            config.method.expert_legs,
            config.method.expert_markers,
            pos_type=config.method.pos_type,
            vel_type=config.method.vel_type,
            torso_type=config.method.torso_type,
            head_type=config.method.head_type,
            head_wrt=config.method.head_wrt,
        )
        expert_obs = [
            torch.from_numpy(x).float().to(config.device) for x in expert_obs_np
        ]
        demos.extend(expert_obs)

    return demos, to_match, mean_reward


@torch.no_grad()
def get_bc_demos_for(
    morphos: torch.Tensor,
    batch: tuple,
    policy: GaussianPolicy,
    device: Union[str, torch.device],
) -> tuple:
    """
    Returns the behavioural cloning demonstrations (i.e., actions) for the given morphologies.

    Parameters
    ----------
    `morphos` -> the morphologies for which to get the demonstrations.
    `batch` -> the batch of observations from which to get the demonstrations.

    Returns
    -------
    The demonstrations for the given morphologies.
    """

    morpho_size = morphos.shape[1]
    feats_batch = torch.FloatTensor(batch[0]).to(device)
    states_batch = feats_batch[:, :-morpho_size]
    demo_feats_batch = torch.cat([states_batch, morphos], dim=1)
    _, _, demo_actions, _ = policy.sample(demo_feats_batch)
    return (
        None,
        demo_actions,
    ) + ((None,) * 7)


def train_wgan_critic(opt, critic, demos: list, batch, use_transitions=True):
    opt.zero_grad()
    device = demos[0].device

    episode_lengths = [len(ep) for ep in demos]
    correct_inds = []
    len_sum = 0
    for length in episode_lengths:
        correct_inds.append(torch.arange(length - 1) + len_sum)
        len_sum += length

    correct_inds = torch.cat(correct_inds)

    demos = torch.cat(demos, dim=0)
    expert_inds = correct_inds[torch.randint(0, len(correct_inds), (len(batch[0]),))]
    expert_feats = demos[expert_inds]

    if use_transitions:
        expert_feats = torch.cat((expert_feats, demos[expert_inds + 1]), dim=1)

    _, _, _, _, _, _, marker_samples, next_marker_samples, _, _ = batch
    marker_feats = marker_samples

    if use_transitions:
        marker_feats = np.concatenate((marker_feats, next_marker_samples), axis=1)

    marker_samples = torch.as_tensor(marker_feats).float().to(device)

    assert expert_feats.shape == marker_samples.shape

    expert_scores = critic(expert_feats)
    policy_scores = critic(marker_samples)

    alpha = torch.rand(marker_samples.shape[0], 1, device=device)

    interpolates = alpha * expert_feats + ((1 - alpha) * marker_samples)
    interpolates.requires_grad_()
    interp_scores = critic(interpolates)

    lamb = 10
    gradients = torch.autograd.grad(
        outputs=interp_scores,
        inputs=interpolates,
        grad_outputs=torch.ones(len(interpolates), 1, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    loss = (policy_scores - expert_scores).mean() + lamb * gradient_penalty

    loss.backward()
    opt.step()

    return (
        loss.item(),
        expert_scores.detach().mean().item(),
        policy_scores.detach().mean().item(),
        gradient_penalty.item(),
    )


def train_disc(
    opt,
    disc: Discriminator,
    demos: List[torch.Tensor],
    batch,
    use_transitions=False,
):
    opt.zero_grad()
    disc.train()

    disc_loss = nn.BCEWithLogitsLoss()

    batch_size = len(batch[0])
    device = demos[0].device

    episode_lengths = [len(ep) for ep in demos]
    correct_inds = []
    len_sum = 0
    for length in episode_lengths:
        correct_inds.append(torch.arange(length - 1) + len_sum)
        len_sum += length

    correct_inds = torch.cat(correct_inds)

    demos = torch.cat(demos, dim=0)
    expert_inds = correct_inds[torch.randint(0, len(correct_inds), (batch_size,))]
    expert_feats = demos[expert_inds]

    if use_transitions:
        expert_feats = torch.cat((expert_feats, demos[expert_inds + 1]), dim=1)

    _, _, _, _, _, _, marker_samples, next_marker_samples, _, _ = batch
    marker_feats = marker_samples

    if use_transitions:
        marker_feats = np.concatenate((marker_feats, next_marker_samples), axis=1)

    marker_samples = torch.as_tensor(marker_feats).float().to(device)

    assert expert_feats.shape == marker_samples.shape

    expert_disc = disc(expert_feats)
    policy_disc = disc(marker_samples)

    expert_labels = 0.8 * torch.ones(
        len(expert_feats), 1, device=demos.device
    )  # NOTE: Shouldn't expert labels be all ones?
    policy_labels = 0.2 * torch.ones(
        len(marker_feats), 1, device=demos.device
    )  # NOTE: Shouldn't policy labels be all zeros?
    # NOTE: This is so the discriminator learns to distinguish between expert and policy samples

    loss = disc_loss(expert_disc, expert_labels) + disc_loss(policy_disc, policy_labels)

    loss.backward()
    opt.step()

    return (
        loss.item(),
        expert_disc.sigmoid().detach().mean().item(),
        policy_disc.sigmoid().detach().mean().item(),
    )
