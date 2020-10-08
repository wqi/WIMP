import torch
import torch.nn as nn


def l1_ewta_loss(prediction, target, k=6, eps=1e-7, mr=2.0):
    num_mixtures = prediction.shape[1]

    target = target.unsqueeze(1).expand(-1, num_mixtures, -1, -1)
    l1_loss = nn.functional.l1_loss(prediction, target, reduction='none').sum(dim=[2, 3])

    # Get loss from top-k mixtures for each timestep
    mixture_loss_sorted, mixture_ranks = torch.sort(l1_loss, descending=False)
    mixture_loss_topk = mixture_loss_sorted.narrow(1, 0, k)

    # Aggregate loss across timesteps and batch
    loss = mixture_loss_topk.sum()
    loss = loss / target.size(0)
    loss = loss / target.size(2)
    loss = loss / k
    return loss


def l1_ewta_loss_prob(prediction, target, k=6, eps=1e-6, mr=2.0):
    num_mixtures = prediction.shape[1]
    output_dim = target.shape[-1]

    target = target.unsqueeze(1).expand(-1, num_mixtures, -1, -1)
    xy_points = prediction.narrow(-1, 0, output_dim)
    probs = prediction.narrow(-1, output_dim, 1).squeeze(-1)
    sequence_probs = nn.functional.softmax(torch.sum(probs, -1), -1)
    sequence_probs_clamped = torch.clamp(sequence_probs, min=0. + eps, max=1. - eps)

    l1_loss = nn.functional.l1_loss(xy_points, target, reduction='none').sum(dim=[2, 3])

    # Get loss from top-k mixtures for each timestep
    mixture_loss_sorted, mixture_ranks = torch.sort(l1_loss, descending=False)
    mixture_loss_topk = mixture_loss_sorted.narrow(1, 0, k)

    # Add probability loss
    selected_mixtures = mixture_ranks.narrow(1, 0, k)
    prob_labels = torch.zeros_like(sequence_probs_clamped)
    prob_labels = prob_labels.scatter(1, selected_mixtures, 1. / k)
    prob_loss = nn.functional.binary_cross_entropy(sequence_probs_clamped, prob_labels)

    # Aggregate loss across timesteps and batch
    loss = mixture_loss_topk.sum()
    loss = loss / target.size(0)
    loss = loss / target.size(2)
    loss = loss / k

    loss = loss + prob_loss
    return loss


def l1_ewta_waypoint_loss(prediction, target, k=6, waypoint_step=5, eps=1e-7):
    num_mixtures = prediction.shape[1]
    timesteps = target.shape[1]

    target = nn.functional.pad(target, pad=(0, 0, 0, waypoint_step - 1))
    target = target.unsqueeze(1).expand(-1, num_mixtures, -1, -1)
    indexes = torch.arange(timesteps).to(target.get_device()) + waypoint_step - 1
    indexes_mask = indexes < timesteps
    indexes_mask = indexes_mask.float()
    curr_label = target.index_select(2, indexes)
    curr_loss_all = nn.functional.l1_loss(prediction.squeeze(-2), curr_label, reduction='none')
    curr_loss_all = curr_loss_all * indexes_mask.view(1, 1, -1, 1)
    curr_loss = curr_loss_all.sum(dim=[2, 3])
    curr_mixture_loss_sorted, curr_mixture_ranks = torch.sort(curr_loss, descending=False)
    curr_mixture_loss_topk = curr_mixture_loss_sorted.narrow(1, 0, k)
    l1_loss = curr_mixture_loss_topk.sum()
    l1_loss = l1_loss / target.size(0)
    l1_loss = l1_loss / (indexes_mask.sum())
    l1_loss = l1_loss / k
    return l1_loss


def l1_ewta_encoder_waypoint_loss(prediction, target, k=6, waypoint_step=5, eps=1e-7, mask=None):
    timesteps = target.shape[1]
    indexes = torch.arange(waypoint_step).to(target.get_device())
    curr_label = target.unsqueeze(1).index_select(2, indexes)
    curr_loss_all = nn.functional.l1_loss(prediction, curr_label, reduction='none')
    if mask is not None:
        curr_mask = mask.index_select(2, indexes)
        curr_loss_all = curr_loss_all * curr_mask.unsqueeze(-1)
    l1_loss = curr_loss_all.sum()
    if mask is None:
        l1_loss = l1_loss / curr_label.size(0)
        l1_loss = l1_loss / curr_label.size(1)
        l1_loss = l1_loss / curr_label.size(2)
    else:
        l1_loss = l1_loss / curr_mask.sum()
    return l1_loss
