from typing import List

import numpy as np
import torch
from torch import nn

from common.models import Discriminator


def train_wgan_critic(opt, critic, expert_obs: list, batch, use_transitions=True):
    opt.zero_grad()
    device = expert_obs[0].device

    episode_lengths = [len(ep) for ep in expert_obs]
    correct_inds = []
    len_sum = 0
    for length in episode_lengths:
        correct_inds.append(torch.arange(length - 1) + len_sum)
        len_sum += length

    correct_inds = torch.cat(correct_inds)

    expert_obs = torch.cat(expert_obs, dim=0)
    expert_inds = correct_inds[torch.randint(0, len(correct_inds), (len(batch[0]),))]
    expert_feats = expert_obs[expert_inds]

    if use_transitions:
        expert_feats = torch.cat((expert_feats, expert_obs[expert_inds + 1]), dim=1)

    _, _, _, _, _, _, marker_samples, next_marker_samples = batch
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
    expert_obs: List[torch.Tensor],
    batch,
    use_transitions=False,
):
    opt.zero_grad()
    disc.train()

    disc_loss = nn.BCEWithLogitsLoss()

    batch_size = len(batch[0])
    device = expert_obs[0].device

    episode_lengths = [len(ep) for ep in expert_obs]
    correct_inds = []
    len_sum = 0
    for length in episode_lengths:
        correct_inds.append(torch.arange(length - 1) + len_sum)
        len_sum += length

    correct_inds = torch.cat(correct_inds)

    expert_obs = torch.cat(expert_obs, dim=0)
    expert_inds = correct_inds[torch.randint(0, len(correct_inds), (batch_size,))]
    expert_feats = expert_obs[expert_inds]

    if use_transitions:
        expert_feats = torch.cat((expert_feats, expert_obs[expert_inds + 1]), dim=1)

    _, _, _, _, _, _, marker_samples, next_marker_samples, _ = batch
    marker_feats = marker_samples

    if use_transitions:
        marker_feats = np.concatenate((marker_feats, next_marker_samples), axis=1)

    marker_samples = torch.as_tensor(marker_feats).float().to(device)

    assert expert_feats.shape == marker_samples.shape

    expert_disc = disc(expert_feats)
    policy_disc = disc(marker_samples)

    expert_labels = 0.8 * torch.ones(
        len(expert_feats), 1, device=expert_obs.device
    )  # NOTE: Shouldn't expert labels be all ones?
    policy_labels = 0.2 * torch.ones(
        len(marker_feats), 1, device=expert_obs.device
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
