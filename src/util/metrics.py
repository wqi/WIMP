import numpy as np
import torch

def compute_metrics(preds, truth, mode='mean', aggregation='agent', miss_threshold=2.0, mask=None):
    """Compute the required evaluation metrics: ADE, FDE, and MR
        Args:
            preds (tensor) [n, a, m, l, d]: predicted trajectories
            truth (tensor) [n, a, l, d]: ground truth trajectory
            mode (string): type of computation to perform
            aggregation (string): whether metrics are computed per-agent or per-scenario
            miss_threshold (float): error threshold in meters to count a prediction as a miss
            mask (tensor) [n, a]: indicator tensor to identify prediction agents
        Returns:
            ade (float): Average Displacement Error
            fde (float): Final Displacement Error
            mr (float): Miss Rate
    """
    # Reshape inputs based on aggregation type
    if aggregation == 'agent':
        truth = truth.unsqueeze(1)
        mask = (mask.sum(-1) > 0).float()
    elif aggregation == 'scenario':
        truth = truth.unsqueeze(1)
        mask = (mask.sum(-1) > 0).float()
    
    # Compute metrics for all agents and scenarios
    l2_all = torch.sqrt(torch.sum((preds - truth)**2, dim=-1))
    ade_all = torch.sum(l2_all, dim=-1) / preds.size(-2)
    fde_all = l2_all[..., -1]
    min_fde_idx = torch.argmin(fde_all, dim=-1).unsqueeze(-1)
    fde = torch.gather(fde_all, -1, min_fde_idx).squeeze(-1)
    ade = torch.gather(ade_all, -1, min_fde_idx).squeeze(-1)
    miss = (fde > miss_threshold).float()

    # Aggregate metrics across agents or scenarios
    if aggregation == 'scenario':
        import ipdb; ipdb.set_trace()
        if mode == 'mean':
            agents_per_scenario = torch.sum(mask, 1)
            fde = torch.mean(torch.sum(fde, 1) / agents_per_scenario)
            ade = torch.mean(torch.sum(ade, 1) / agents_per_scenario)
            mr = torch.sum(miss) / mask.sum()  # Reported per-agent
        elif mode == 'worst':
            wc_fde = torch.max(fde, 1)[0]
            wc_ade = torch.max(ade, 1)[0]
            fde = torch.mean(wc_fde)
            ade = torch.mean(wc_ade)
            mr = torch.mean((wc_fde > miss_threshold).float())  # Reported per-scenario
    elif aggregation == 'agent':
        fde = torch.sum(fde) / mask.sum()
        ade = torch.sum(ade) / mask.sum()
        mr = torch.sum(miss) / mask.sum()

    return (ade, fde, mr)

def compute_metrics_1(prediction, truth, mean=True, on_gpu=True, miss_threshold=2.0):
    """Compute the required evaluation metrics: ADE, FDE, and MR
        Args:
            prediction (array): predicted trajectories
            truth (array): ground truth trajectory
        Returns:
            ade (float): Average Displacement Error
            fde (float): Final Displacement Error
            mr (float): Miss Rate
    """
    if on_gpu:
        truth = truth.unsqueeze(1)
        l2_all = torch.sqrt(torch.sum((prediction - truth)**2, dim=-1))
        ade_all = torch.sum(l2_all, dim=-1) / prediction.size(-2)
        fde_all = l2_all[..., -1]
        min_fde = torch.argmin(fde_all, dim=-1)
        indices = torch.arange(prediction.shape[0], device=min_fde.get_device())
        fde = fde_all[indices, min_fde]
        ade = ade_all[indices, min_fde]
        miss = (fde > miss_threshold).float()
        if mean:
            return torch.mean(ade), torch.mean(fde), torch.mean(miss)
        else:
            return ade, fde, miss
    else:
        truth = np.expand_dims(truth, 1)
        l2_all = np.sqrt(np.sum((prediction - truth)**2, axis=-1))
        ade_all = np.sum(l2_all, axis=-1) / prediction.shape[-2]
        fde_all = l2_all[..., -1]
        min_fde = np.argmin(fde_all, axis=-1)
        indices = np.arange(prediction.shape[0])
        fde = fde_all[indices, min_fde]
        ade = ade_all[indices, min_fde]
        miss = (fde > miss_threshold).astype(np.float32)
        if mean:
            return np.mean(ade), np.mean(fde), np.mean(miss)
        else:
            return ade, fde, miss
