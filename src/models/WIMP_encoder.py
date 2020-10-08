import numpy as np
import torch
import torch.nn as nn


class WIMPEncoder(nn.Module):
    def __init__(self, hparams):
        super(WIMPEncoder, self).__init__()
        self.hparams = hparams
        self.hparams.cl_kernel_list = [1, 3, 5]
        self.hparams.xy_kernel_list = [1, 3, 5]

        # Define Modules
        self.xy_conv_filters = nn.ModuleList([nn.Conv1d(in_channels=self.hparams.input_dim,
                                              out_channels=self.hparams.hidden_dim,
                                              kernel_size=x, padding=(x - 1) // 2) for x
                                              in self.hparams.xy_kernel_list])
        self.xy_input_transform = nn.Conv1d(in_channels=self.hparams.hidden_dim * len(self.hparams.xy_kernel_list),
                                            out_channels=self.hparams.hidden_dim, kernel_size=1)
        self.non_linearity = nn.Tanh() if self.hparams.non_linearity == 'tanh' else nn.ReLU()
        self.centerline_modifier = 2 if \
            (self.hparams.use_centerline_features and not self.hparams.add_centerline) else 1
        self.lstm_input_transform = nn.Linear(self.hparams.hidden_dim * self.centerline_modifier,
                                              self.hparams.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.hparams.hidden_dim, hidden_size=self.hparams.hidden_dim,
                            num_layers=self.hparams.num_layers, batch_first=True,
                            dropout=self.hparams.dropout)
        self.waypoint_predictor = nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim) if \
            self.hparams.output_prediction else \
            nn.Linear(self.hparams.hidden_dim * self.hparams.num_layers, self.hparams.output_dim)
        self.waypoint_lstm = nn.LSTM(input_size=self.hparams.hidden_dim,
                                     hidden_size=self.hparams.hidden_dim,
                                     num_layers=self.hparams.num_layers, batch_first=True,
                                     dropout=self.hparams.dropout)
        if self.hparams.use_centerline_features:
            key_input = self.hparams.hidden_dim if not self.hparams.hidden_key_generator else \
                self.hparams.hidden_dim * self.hparams.num_layers
            self.cl_conv_filters = nn.ModuleList([nn.Conv1d(in_channels=self.hparams.input_dim,
                                                  out_channels=self.hparams.hidden_dim,
                                                  kernel_size=x, padding=(x - 1) // 2)
                                                  for x in self.hparams.cl_kernel_list])
            self.cl_input_transform = nn.Conv1d(in_channels=self.hparams.hidden_dim * len(self.hparams.cl_kernel_list), out_channels=self.hparams.hidden_dim, kernel_size=1)
            self.leakyrelu = nn.LeakyReLU()
            self.key_generator = nn.Linear(key_input, self.hparams.hidden_dim)
            self.query_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
            self.value_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)

        if self.hparams.batch_norm:
            self.input_bn = nn.BatchNorm1d(self.hparams.hidden_dim)

    def forward(self, agent_features, social_features, num_agent_mask, ifc_helpers=None, visualize_centerline=False):
        non_zero_indices = torch.nonzero(num_agent_mask.view(-1,), as_tuple=True)[0]
        zero_indices = torch.nonzero(num_agent_mask.view(-1,) == 0, as_tuple=True)[0]

        if self.hparams.use_centerline_features:
            agent_centerline = ifc_helpers['agent_oracle_centerline']
            agent_centerline_lengths = ifc_helpers['agent_oracle_centerline_lengths']
            social_centerline = ifc_helpers['social_oracle_centerline']
            social_centerline_lengths = ifc_helpers['social_oracle_centerline_lengths']

            all_centerline = torch.cat([agent_centerline.unsqueeze(1), social_centerline], dim=1).view(-1, *agent_centerline.size()[1:])
            all_centerline_nonzero = all_centerline.index_select(0, non_zero_indices)
            all_centerline_nonzero_transposed = all_centerline_nonzero.transpose(1,2).contiguous()
            all_centerline_lengths = torch.cat([agent_centerline_lengths.unsqueeze(1), social_centerline_lengths], dim=1).view(-1,)
            all_centerline_lengths_nonzero = all_centerline_lengths.index_select(0, non_zero_indices)
            all_centerline_features = []
            for i, _ in enumerate(self.hparams.cl_kernel_list):
                all_centerline_features.append(self.cl_conv_filters[i](all_centerline_nonzero_transposed))
            all_centerline_features = torch.cat(all_centerline_features, dim=1)
            all_centerline_features_comb = self.non_linearity(self.cl_input_transform(all_centerline_features))
            centerline_features = all_centerline_features_comb.transpose(1,2).contiguous()

            with torch.no_grad():
                indexer = torch.arange(centerline_features.size(1)).type_as(centerline_features)
                centerline_mask_byte = all_centerline_lengths_nonzero[:, None] > indexer
                centerlines_masked = torch.where(centerline_mask_byte.unsqueeze(-1), all_centerline_nonzero, torch.zeros_like(all_centerline_nonzero).fill_(np.float("inf")))

        if self.hparams.distributed_backend == 'dp':
            self.lstm.flatten_parameters()

        all_agents = torch.cat([agent_features.unsqueeze(1), social_features], dim=1).view(-1, *agent_features.size()[1:])
        all_agents_nonzero = all_agents.index_select(0, non_zero_indices)
        _ , resorter = torch.sort(torch.cat([non_zero_indices, zero_indices], dim=0), descending=False)
        all_agents_nonzero_transposed = all_agents_nonzero.transpose(1,2).contiguous()

        # Compute convolution filters and encoding
        conv_filters = []
        for i, _ in enumerate(self.hparams.xy_kernel_list):
            conv_filters.append(self.xy_conv_filters[i](all_agents_nonzero_transposed))
        conv_filters = torch.cat(conv_filters, dim=1)
        input_features = self.non_linearity(self.xy_input_transform(conv_filters))
        input_features = input_features.transpose(1,2).contiguous()
        if self.hparams.batch_norm:
            input_features = self.input_bn(input_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        # Initialize Hidden State
        hidden = self.initHidden(self.hparams.num_layers, input_features.size(0))
        if visualize_centerline:
            centerline_attention_viz = []

        # Iterate over all input timesteps
        waypoint_predictions = []
        centerline_features_query = self.query_generator(centerline_features)
        centerline_features_value = self.value_generator(centerline_features)
        for tstep in range(input_features.size(1)):
            curr_waypoint_points = []
            current_input = input_features.narrow(1, start=tstep, length=1)
            with torch.no_grad():
                # Find closest point on centerline from predicted waypoint
                curr_xy_points = all_agents_nonzero.narrow(1, tstep, 1).detach()
                distances = centerlines_masked - curr_xy_points
                distances = torch.sum(torch.mul(distances, distances), dim=-1)
                closest_point = torch.argmin(distances, dim=-1)

            # Waypoint Prediction
            if tstep < (input_features.size(1) - self.hparams.waypoint_step):
                curr_waypoint_prediction = all_agents_nonzero.narrow(1, tstep + self.hparams.waypoint_step, 1).detach()
            else:
                curr_waypoint_features = current_input.view(current_input.size(0), 1, -1)
                curr_waypoint_decoding, curr_waypoint_hidden = self.waypoint_lstm(curr_waypoint_features, hidden)
                # Make prediction
                if self.hparams.output_prediction:
                    curr_waypoint_prediction = self.waypoint_predictor(curr_waypoint_decoding)
                else:
                    curr_waypoint_prediction = self.waypoint_predictor(curr_waypoint_hidden[0].transpose(0,1).contiguous().view(curr_waypoint_hidden[0].size(1), -1)).unsqueeze(1)
                curr_waypoint_points.append(curr_waypoint_prediction)
                curr_waypoint_points = torch.stack(curr_waypoint_points,1)
                waypoint_predictions.append(curr_waypoint_points)
            with torch.no_grad():
                # Find closest point on centerline from predicted waypoint
                curr_xy_points = all_agents_nonzero.narrow(1, tstep, 1).detach()
                distances = centerlines_masked - curr_xy_points
                distances = torch.sum(torch.mul(distances, distances), dim=-1)
                closest_point = torch.argmin(distances, dim=-1)

                waypoint_distances = centerlines_masked - curr_waypoint_prediction
                waypoint_distances = torch.sum(torch.mul(waypoint_distances, waypoint_distances), dim=-1)
                waypoint_closest_point = torch.argmin(waypoint_distances, dim=-1)
                segment_length = waypoint_closest_point - closest_point

                max_length = segment_length.abs().max().data.cpu().numpy()
                arange_array = torch.arange(int(max_length) + 1).type_as(segment_length)
                upper_array = closest_point.unsqueeze(-1) + arange_array.view(1,-1)
                lower_array = closest_point.unsqueeze(-1) - arange_array.view(1,-1)

                positive_length_mask = segment_length >= 0
                indexing_array = torch.where(positive_length_mask.unsqueeze(-1).expand(-1,upper_array.size(-1)), upper_array, lower_array)
                positive_mask = indexing_array <= waypoint_closest_point.unsqueeze(-1)
                negative_mask = indexing_array >= waypoint_closest_point.unsqueeze(-1)
                indexing_mask = torch.where(positive_length_mask.unsqueeze(-1).byte(), positive_mask.byte(), negative_mask.byte())

                # Check if indexing array is less than 0
                lower_mask = indexing_array < 0
                upper_mask = indexing_array >= centerlines_masked.size(1)
                indexing_array[lower_mask] = 0
                indexing_array[upper_mask] = centerlines_masked.size(1) - 1

            curr_centerline_features = torch.gather(centerline_features_query, 1, indexing_array.unsqueeze(-1).expand(-1,-1,self.hparams.hidden_dim))  
            curr_centerline_features_value = torch.gather(centerline_features_value, 1, indexing_array.unsqueeze(-1).expand(-1,-1,self.hparams.hidden_dim))
            # Compute attention over centerline segments
            current_key = self.key_generator(current_input) if not self.hparams.hidden_key_generator else self.key_generator(hidden[0].transpose(0,1).contiguous().view(current_input.size(0), 1, -1))
            current_centerline_score_unnormalized = nn.functional.leaky_relu(torch.bmm(curr_centerline_features , current_key.transpose(1,2)))
            current_centerline_score_unnormalized = current_centerline_score_unnormalized.view(*indexing_mask.size())
            current_centerline_score = torch.where(indexing_mask, current_centerline_score_unnormalized, torch.zeros_like(current_centerline_score_unnormalized).fill_(np.float("-inf")))
            current_centerline_attention = nn.functional.softmax(current_centerline_score, -1)
            curr_centerline = curr_centerline_features_value * current_centerline_attention.unsqueeze(-1)
            curr_centerline = torch.sum(curr_centerline, dim=1, keepdim=True)
            current_input_xy_centerline = torch.cat([current_input, curr_centerline], dim=-1) if not self.hparams.add_centerline else (current_input + curr_centerline)
            current_input_xy_centerline = self.non_linearity(self.lstm_input_transform(current_input_xy_centerline))
            # if self.hparams.batch_norm:
            #     current_input_xy_centerline = self.input_bn(current_input_xy_centerline.transpose(1,2).contiguous()).transpose(1,2).contiguous()
            current_encoding, hidden = self.lstm(current_input_xy_centerline, hidden)

        # Pad zeros for dummy agents
        waypoint_predictions_nonzero = torch.cat(waypoint_predictions, dim=1)
        waypoint_predictions_pad = nn.functional.pad(waypoint_predictions_nonzero, pad=(0,0,0,0,0,0,0,all_agents.size(0) - all_agents_nonzero.size(0)))
        waypoint_predictions_all = waypoint_predictions_pad.index_select(0, resorter)
        current_encoding_pad = nn.functional.pad(current_encoding, pad=(0,0,0,0,0,all_agents.size(0) - all_agents_nonzero.size(0)))
        current_encoding = current_encoding_pad.index_select(0, resorter)
        hidden_pad = (nn.functional.pad(hidden[0], pad=(0,0,0,all_agents.size(0) - all_agents_nonzero.size(0),0,0)), nn.functional.pad(hidden[1], pad=(0,0,0,all_agents.size(0) - all_agents_nonzero.size(0),0,0)))
        hidden = (hidden_pad[0].index_select(1, resorter), hidden_pad[1].index_select(1, resorter))
        return current_encoding, hidden, waypoint_predictions_all

    def initHidden(self, batch_size=1, num_agents=1):
        weight = next(self.parameters()).data
        return (weight.new(batch_size, num_agents, self.hparams.hidden_dim).zero_(), weight.new(batch_size, num_agents, self.hparams.hidden_dim).zero_())
