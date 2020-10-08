import numpy as np
import torch
import torch.nn as nn


class WIMPDecoder(nn.Module):
    def __init__(self, hparams):
        super(WIMPDecoder, self).__init__()
        self.hparams = hparams
        self.hparams.cl_selected_kernel_list = [1, 3]
        self.hparams.output_xy_kernel_list = self.hparams.xy_kernel_list if self.hparams.output_conv else [1]
        self.hparams.predictor_output_dim = self.hparams.num_mixtures*(self.hparams.output_dim + 1)

        self.xy_conv_filters = nn.ModuleList([nn.Conv1d(in_channels=self.hparams.input_dim + 1, out_channels=self.hparams.hidden_dim, kernel_size=x) for x in self.hparams.output_xy_kernel_list])
        self.output_transform = nn.Conv1d(in_channels=self.hparams.hidden_dim*len(self.hparams.output_xy_kernel_list), out_channels=self.hparams.hidden_dim, kernel_size=1)

        self.non_linearity = nn.Tanh() if self.hparams.non_linearity == 'tanh' else nn.ReLU()
        self.lstm = nn.LSTM(input_size=self.hparams.hidden_dim*self.hparams.num_mixtures, hidden_size=self.hparams.hidden_dim, num_layers=self.hparams.num_layers, batch_first=True, dropout=self.hparams.dropout)
        self.centerline_modifier = 2 if (self.hparams.use_centerline_features and not self.hparams.add_centerline) else 1
        self.lstm_input_transform = nn.Linear(self.hparams.hidden_dim*self.centerline_modifier, self.hparams.hidden_dim)
        self.predictor = nn.Linear(self.hparams.hidden_dim, self.hparams.predictor_output_dim) if self.hparams.output_prediction else nn.Linear(self.hparams.hidden_dim*self.hparams.num_layers, self.hparams.predictor_output_dim) 
        self.waypoint_predictor = nn.Linear(self.hparams.hidden_dim, self.hparams.predictor_output_dim - self.hparams.num_mixtures) if self.hparams.output_prediction else nn.Linear(self.hparams.hidden_dim*self.hparams.num_layers, self.hparams.predictor_output_dim) 
        self.waypoint_lstm = nn.LSTM(input_size=self.hparams.hidden_dim*self.hparams.num_mixtures, hidden_size=self.hparams.hidden_dim, num_layers=self.hparams.num_layers, batch_first=True, dropout=self.hparams.dropout)
        if self.hparams.use_centerline_features:
            key_input = self.hparams.hidden_dim if not self.hparams.hidden_key_generator else self.hparams.hidden_dim*self.hparams.num_layers
            self.cl_conv_filters = nn.ModuleList([nn.Conv1d(in_channels=self.hparams.input_dim, out_channels=self.hparams.hidden_dim, kernel_size=x, padding=(x-1)//2) for x in self.hparams.cl_kernel_list])
            self.cl_input_transform = nn.Conv1d(in_channels=self.hparams.hidden_dim*len(self.hparams.cl_kernel_list), out_channels=self.hparams.hidden_dim, kernel_size=1)
            self.leakyrelu = nn.LeakyReLU()
            key_output = self.hparams.hidden_dim if not self.hparams.hidden_key_generator else self.hparams.hidden_dim*self.hparams.num_mixtures
            self.key_generator = nn.Linear(key_input, key_output)
            self.query_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
            self.value_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)

    def forward(self, decoder_input_features, last_n_predictions, hidden_decoder, outsteps, ifc_helpers=None, sample_next=False, map_estimate=False, mixture_num=-1, sample_centerline=False):
        if self.hparams.use_centerline_features:
            agent_centerline = ifc_helpers['agent_oracle_centerline']
            agent_centerline_lengths = ifc_helpers['agent_oracle_centerline_lengths']
            agent_centerline = agent_centerline.transpose(1,2).contiguous()
            agent_centerline_features = []
            for i, _ in enumerate(self.hparams.cl_kernel_list):
                agent_centerline_features.append(self.cl_conv_filters[i](agent_centerline))
            agent_centerline_features = torch.cat(agent_centerline_features, dim=1)
            agent_centerline_features_comb = self.non_linearity(self.cl_input_transform(agent_centerline_features))
            centerline_features = agent_centerline_features_comb.transpose(1,2).contiguous()
            selected_centerline_features = centerline_features.unsqueeze(1).repeat(1, self.hparams.num_mixtures, 1, 1)
            selected_centerlines = ifc_helpers['agent_oracle_centerline'].unsqueeze(1).repeat(1, self.hparams.num_mixtures, 1, 1)
            selected_centerline_lengths = agent_centerline_lengths.unsqueeze(1).repeat(1, self.hparams.num_mixtures)
            with torch.no_grad():
                indexer = torch.arange(selected_centerlines.size(2)).to(selected_centerlines.get_device())
                centerline_mask_byte = selected_centerline_lengths[:,:,None] > indexer
                centerline_mask = centerline_mask_byte.float()
                centerline_mask_nonzero_indexer = torch.nonzero(centerline_mask.view(-1), as_tuple=True)[0]
                centerline_mask_zero_indexer = torch.nonzero(centerline_mask.view(-1)==0, as_tuple=True)[0]
                _, centerline_resorter = torch.sort(torch.cat([centerline_mask_nonzero_indexer, centerline_mask_zero_indexer], dim=0), descending=False)
                selected_centerlines_masked = torch.where(centerline_mask_byte.unsqueeze(-1), selected_centerlines, torch.zeros_like(selected_centerlines).fill_(np.float("inf")))

        if self.hparams.distributed_backend == 'dp':
            self.lstm.flatten_parameters()
            self.waypoint_lstm.flatten_parameters()

        # Decode
        predictions = []
        waypoint_predictions = []
        num_batches = decoder_input_features.size(0)
        for i, _ in enumerate(self.hparams.output_xy_kernel_list):
            last_n_predictions[i] = nn.functional.pad(last_n_predictions[i], pad=(0,1), value=1.0)
            last_n_predictions[i] = last_n_predictions[i].unsqueeze(1).repeat(1,self.hparams.num_mixtures,1,1).view(-1, *last_n_predictions[i].size()[1:])
        decoder_input_features = nn.functional.pad(decoder_input_features, pad=(0,1), value=1.0)
        decoder_input_features = decoder_input_features.unsqueeze(1).repeat(1,self.hparams.num_mixtures,1,1).view(-1, *decoder_input_features.size()[1:])
        selected_centerline_features_query = self.query_generator(selected_centerline_features)
        selected_centerline_features_value = self.value_generator(selected_centerline_features)
        for timestep in range(outsteps):
            curr_conv_filters = []
            curr_waypoint_points = []
            if self.hparams.output_conv:
                for i, _ in enumerate(self.hparams.output_xy_kernel_list):
                    curr_conv_filters.append(self.xy_conv_filters[i](last_n_predictions[i].transpose(1,2).contiguous()).view(num_batches,self.hparams.num_mixtures,-1))
            else:
                for i, _ in enumerate(self.hparams.output_xy_kernel_list):
                    curr_conv_filters.append(self.xy_conv_filters[i](decoder_input_features.transpose(1,2).contiguous()).view(num_batches,self.hparams.num_mixtures,-1))
            curr_conv_filters = torch.cat(curr_conv_filters, dim=2).transpose(1,2).contiguous()
            curr_features = self.non_linearity(self.output_transform(curr_conv_filters))
            curr_features = curr_features.transpose(1,2).contiguous()

            # Waypoint Prediction
            curr_waypoint_features = curr_features.clone().view(curr_features.size(0), 1, -1)
            curr_waypoint_decoding, curr_waypoint_hidden = self.waypoint_lstm(curr_waypoint_features, hidden_decoder)
            # Make prediction
            if self.hparams.output_prediction:
                curr_waypoint_prediction = self.waypoint_predictor(curr_waypoint_decoding)
            else:
                curr_waypoint_prediction = self.waypoint_predictor(curr_waypoint_hidden[0].transpose(0,1).contiguous().view(curr_waypoint_hidden[0].size(1), -1)).unsqueeze(1)
            curr_waypoint_prediction = curr_waypoint_prediction.view(curr_waypoint_prediction.size(0), self.hparams.num_mixtures, -1)
            curr_waypoint_points.append(curr_waypoint_prediction)
            with torch.no_grad():
                # Find closest point on centerline from predicted waypoint
                curr_xy_points = decoder_input_features.view(num_batches, self.hparams.num_mixtures, 1, -1).narrow(-1, 0, self.hparams.input_dim)
                distances = selected_centerlines_masked - curr_xy_points
                distances = torch.sum(torch.mul(distances, distances), dim=-1)
                closest_point = torch.argmin(distances, dim=-1)

                waypoint_distances = selected_centerlines_masked - curr_waypoint_prediction.unsqueeze(2)
                waypoint_distances = torch.sum(torch.mul(waypoint_distances, waypoint_distances), dim=-1)
                waypoint_closest_point = torch.argmin(waypoint_distances, dim=-1)

                segment_length = waypoint_closest_point - closest_point
                max_length = segment_length.abs().max().data.cpu().numpy()
                arange_array = torch.arange(int(max_length) + 1).cuda()
                upper_array = closest_point.unsqueeze(-1) + arange_array.view(1,1,-1)
                lower_array = closest_point.unsqueeze(-1) - arange_array.view(1,1,-1)
                positive_length_mask = segment_length >= 0
                indexing_array = torch.where(positive_length_mask.unsqueeze(-1).expand(-1,-1,upper_array.size(-1)), upper_array, lower_array)
                positive_mask = indexing_array <= waypoint_closest_point.unsqueeze(-1)
                negative_mask = indexing_array >= waypoint_closest_point.unsqueeze(-1)
                indexing_mask = torch.where(positive_length_mask.unsqueeze(-1), positive_mask, negative_mask)

                # Check if indexing array is less than 0
                lower_mask = indexing_array < 0
                upper_mask = indexing_array >= selected_centerlines_masked.size(2)
                indexing_array[lower_mask] = 0
                indexing_array[upper_mask] = selected_centerlines_masked.size(2) - 1

            curr_centerline_features = torch.gather(selected_centerline_features_query, 2, indexing_array.unsqueeze(-1).expand(-1,-1,-1,self.hparams.hidden_dim))
            curr_centerline_features_value = torch.gather(selected_centerline_features_value, 2, indexing_array.unsqueeze(-1).expand(-1,-1,-1,self.hparams.hidden_dim))
            # Compute attention over centerline segments
            current_key = self.key_generator(curr_features) if not self.hparams.hidden_key_generator else self.key_generator(hidden_decoder[0].transpose(0,1).contiguous().view(curr_features.size(0), 1, -1))
            if self.hparams.hidden_key_generator:
                current_key = current_key.view(-1, self.hparams.num_mixtures, self.hparams.hidden_dim)
            current_centerline_score_unnormalized = nn.functional.leaky_relu(torch.bmm(curr_centerline_features.view(-1, *curr_centerline_features.size()[2:]) , current_key.view(-1, 1, current_key.size(2)).transpose(1,2)))
            current_centerline_score_unnormalized = current_centerline_score_unnormalized.view(*indexing_mask.size())
            current_centerline_score = torch.where(indexing_mask, current_centerline_score_unnormalized, torch.zeros_like(current_centerline_score_unnormalized).fill_(np.float("-inf")))
            current_centerline_attention = nn.functional.softmax(current_centerline_score, -1)
            curr_centerline = curr_centerline_features_value * current_centerline_attention.unsqueeze(-1)
            curr_centerline = torch.sum(curr_centerline, dim=2)
            current_input_xy_centerline = torch.cat([curr_features, curr_centerline], dim=-1) if not self.hparams.add_centerline else (curr_features + curr_centerline)
            current_input_xy_centerline = self.non_linearity(self.lstm_input_transform(current_input_xy_centerline))
            current_input_xy_centerline = current_input_xy_centerline.view(current_input_xy_centerline.size(0), 1, -1)
            current_decoding, hidden_decoder = self.lstm(current_input_xy_centerline, hidden_decoder)
            if self.hparams.output_prediction:
                curr_prediction = self.predictor(current_decoding)
            else:
                curr_prediction = self.predictor(hidden_decoder[0].transpose(0,1).contiguous().view(hidden_decoder[0].size(1), -1)).unsqueeze(1)
            curr_prediction = curr_prediction.view(curr_prediction.size(0), self.hparams.num_mixtures, -1)
            curr_probs = curr_prediction.narrow(-1, self.hparams.output_dim, 1)
            curr_probs = -1 * torch.relu(curr_probs)
            curr_prob_modified = torch.cat([curr_prediction.narrow(-1, 0, self.hparams.output_dim), curr_probs], -1)
            predictions.append(curr_prob_modified)
            waypoint_predictions.append(torch.stack(curr_waypoint_points,2))
            decoder_input_features = curr_prob_modified.detach().view(*decoder_input_features.size())

            for i, ksize in enumerate(self.hparams.output_xy_kernel_list):
                last_n_predictions[i] = torch.cat([last_n_predictions[i].narrow(dim=1, start=1, length=ksize-1), decoder_input_features], dim=1).detach()

        predictions_tensor = torch.stack(predictions, 2).unsqueeze(1)
        waypoint_tensor = torch.stack(waypoint_predictions, 2).unsqueeze(1)
        return predictions_tensor, waypoint_tensor, []
