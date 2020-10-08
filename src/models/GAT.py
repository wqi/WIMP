import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gat_iters=1, num_heads=4, dropout=0.5,
                 alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gat_iters = num_gat_iters
        self.num_heads = num_heads
        self.alpha = alpha

        # Define and initialize trainable components
        self.W = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_heads)]) # NOQA
        self.a_1 = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(self.num_heads)])
        self.a_2 = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(self.num_heads)])

        # Define other components
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency):
        att_weights = []
        cur_h = h

        for iter in range(self.num_gat_iters):
            head_embeds = []

            for head in range(self.num_heads):
                # Compute attention coefficients
                cur_h_transformed = self.W[head](cur_h)
                att_half_1 = self.a_1[head](cur_h_transformed).squeeze(-1)
                att_half_2 = self.a_2[head](cur_h_transformed).squeeze(-1)
                att_coeff = att_half_1.unsqueeze(-2) + att_half_2.unsqueeze(-3)
                att_coeff = self.leakyrelu(att_coeff)

                # Compute softmax over connected edges using adjacency matrix
                with torch.no_grad():
                    masked_att_max = torch.max(att_coeff, 2)[0]
                masked_att_reduced = att_coeff.squeeze(-1) - masked_att_max
                masked_att_exp = masked_att_reduced.exp() * adjacency
                masked_att_exp = masked_att_exp.unsqueeze(-1)

                mask_sum = masked_att_exp.sum(dim=2, keepdim=True)
                mask_ones = torch.ones_like(mask_sum)
                mask_sum_normalized = torch.where(mask_sum == 0.0, mask_ones, mask_sum)
                att_values = torch.div(masked_att_exp, mask_sum_normalized)
                att_values = self.dropout(att_values)

                # Compute head embedding using learned attention weights
                h_prime = torch.bmm(att_values.squeeze(-1),
                                    cur_h_transformed.squeeze(-2)).unsqueeze(-2)
                head_embeds.append(h_prime)

                # Record attention weights on first GAT iteration
                if iter == 0:
                    att_weights.append(att_values.squeeze(-1).detach())

            # Compute updated embeding using residual
            cur_h = torch.tanh(cur_h + torch.mean(torch.stack(head_embeds, dim=-1), dim=-1))

        out = cur_h
        att_weights = torch.stack(att_weights, dim=1)
        return out, att_weights
