import numpy as np
import os
import pickle
import re
import torch

from collections import defaultdict
from itertools import zip_longest
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import Dataset


class ArgoverseDataset(Dataset):
    """Dataloader for the processed Argoverse forecasting data.
        Args:
            data_loc (string): Location to the processed pickle file. Use preprocess.py to obtain this pickle
            mode (string): Type of dataset: train|test|val
            transform (torchvision.transforms): Tranformation to apply to each example
            delta (bool): Make model predict delta changes instead of absolute values
            xy_features_flag (bool): Use xy coordinate features
            xy_features_normalize_flag (bool): Normalize xy features to constraint start of sequence to be (0,0) and end of sequence to be on the positive x axis
            map_features_flag (bool): Compute map features if true
            social_features_flag (bool): Compute social features if true
            timesteps (int): Timesteps for which feature computation needs to be done (10 timesteps = 1 second)
            filtered_idxs (list[int]): Specific idxs from dataframe to use in dataloader
    """
    def __init__(self, data_loc, mode='train', transform=None, delta=True, timesteps=20,
                 outsteps=30, augment_data=False, map_features_flag=True, social_features_flag=True,
                 heuristic=False, ifc=True, is_oracle=False):
        self.data_loc = data_loc
        self.transform = transform
        self.mode = mode
        self.map_features_flag = map_features_flag
        self.social_features_flag = social_features_flag
        self.heuristic = heuristic
        self.timesteps = timesteps
        self.outsteps = outsteps
        self.is_oracle = is_oracle

        if mode == 'trainval':
            self.sequences = [os.path.join("{}/{}".format(data_loc, 'train'), file) for file in os.listdir("{}/{}".format(data_loc, 'train'))]
            self.sequences = self.sequences + [os.path.join("{}/{}".format(data_loc, 'val'), file) for file in os.listdir("{}/{}".format(data_loc, 'val'))]
        else:
            self.sequences = [os.path.join("{}/{}".format(data_loc, mode), file) for file in os.listdir("{}/{}".format(data_loc, mode))]
        # self.sequences = self.sequences[:100] # TEMP
        self.delta = delta
        self.delta_str = "_delta" if self.delta else ""
        self.str = "_PARTIAL"
        self.heuristic_str = "HEURISTIC_" if self.heuristic else ""
        self.max_social_agents = 30

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Returns a single example from dataset
            Args:
                idx: Index of example
            Returns:
                output: Necessary values for example
        """
        example = {}
        example['idx'] = idx
        example['file_path'] = self.sequences[idx]
        with open(example['file_path'], 'rb') as inFile:
            data = pickle.load(inFile)

        example['seq_id'] = data['SEQ_ID']
        example['city'] = data['CITY_NAME']
        # Get feature helpers
        if 'TRANSLATION' in data:
            example['translation'] = np.array(data['TRANSLATION'])
        if 'ROTATION' in data:
            example['rotation'] = np.array(data['ROTATION'])

        # Get focal agent features
        example['agent_xy_features'] = data['AGENT']['XY_FEATURES']
        if 'LABELS' in data['AGENT']:
            example['agent_xy_labels'] = data['AGENT']['LABELS']
        else:
            example['agent_xy_labels'] = np.zeros((self.outsteps, 2), dtype=np.float)

        agent_str = '_FULL' if self.is_oracle else '_PARTIAL'

        # Get centerline for IFC
        if not self.is_oracle:
            example['agent_oracle_centerline'] = data['AGENT'][self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+agent_str]
            example['agent_oracle_centerline_lengths'] = example['agent_oracle_centerline'].shape[0]

            # Add noise
            if self.mode == 'train':
                rotation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                rotation = np.random.random() * 27.0 * rotation_sign
                translation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                translation = np.random.random(2) * translation_sign
                agent_all_features = np.vstack([example['agent_xy_features'], example['agent_xy_labels']])
                agent_all_features = self.add_noise(agent_all_features, rotation=rotation, translation=translation)
                example['agent_xy_features'] = agent_all_features[:example['agent_xy_features'].shape[0],:]
                example['agent_xy_labels'] = agent_all_features[example['agent_xy_features'].shape[0]:,:]
                example['agent_oracle_centerline'] = self.add_noise(example['agent_oracle_centerline'], rotation=rotation, translation=translation)
        else:

            example['agent_oracle_centerline'] = data['AGENT']['TEST_CANDIDATE_CENTERLINE_NORMALIZED'+agent_str]
            example['agent_oracle_centerline_lengths'] = [x.shape[0] for x in example['agent_oracle_centerline']]

            # Add noise
            if self.mode == 'train':
                rotation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                rotation = np.random.random() * 27.0 * rotation_sign
                translation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                translation = np.random.random(2) * translation_sign
                agent_all_features = np.vstack([example['agent_xy_features'], example['agent_xy_labels']])
                agent_all_features = self.add_noise(agent_all_features, rotation=rotation, translation=translation)
                example['agent_xy_features'] = agent_all_features[:example['agent_xy_features'].shape[0],:]
                example['agent_xy_labels'] = agent_all_features[example['agent_xy_features'].shape[0]:,:]
                example['agent_oracle_centerline'] = [self.add_noise(x, rotation=rotation, translation=translation) for x in example['agent_oracle_centerline']]

            # Pad centerlines
            # max_pad = np.max(example['agent_oracle_centerline_lengths'])
            # for index, elem in enumerate(example['agent_oracle_centerline']):
            #     num_pad = max_pad - elem.shape[0]
            #     padded_elem = np.pad(elem, ((0,num_pad),(0,0)), 'constant', constant_values=(0.,))
            #     example['agent_oracle_centerline'][index] = padded_elem
            # example['agent_oracle_centerline'] = np.array(example['agent_oracle_centerline'])
            # example['agent_oracle_centerline_lengths'] = np.array(example['agent_oracle_centerline_lengths'])

        # Compute delta xy coordinates if required
        if self.delta:
            padded_xy_delta, padded_labels_delta, ref_start, ref_end = self.relative_distance_with_labels(example['agent_xy_features'], example['agent_xy_labels'])
            example['agent_xy_features_delta'] = padded_xy_delta
            example['agent_xy_labels_delta'] = padded_labels_delta
            example['agent_xy_ref_start'] = ref_start
            example['agent_xy_ref_end'] = ref_end

        # Get social agent features
        num_social_agents = 0
        if self.social_features_flag:
            social = defaultdict(list)
            for social_num, social_features in enumerate(data['SOCIAL']):
                if social_num >= self.max_social_agents:
                    break
                tstamps = social_features['TSTAMPS']
                # Check if social agent has 2 seconds of history
                if social_features['XY_FEATURES'].shape[0] == self.timesteps:

                    # Compute mask for agents that don't have information for all timesteps
                    mask = np.full(self.timesteps + self.outsteps, False)
                    mask[tstamps] = True
                    input_mask = mask[:self.timesteps]
                    label_mask = mask[self.timesteps:]
                    social['social_input_mask'].append(input_mask)
                    social['social_label_mask'].append(label_mask)

                    # Add noise
                    if self.mode == 'train':
                        if 'LABELS' in social_features and len(social_features['LABELS']) > 0:
                            all_features = np.vstack([social_features['XY_FEATURES'], social_features['LABELS']])
                        else:
                            all_features = social_features['XY_FEATURES']
                        all_features = self.add_noise(all_features, rotation=rotation, translation=translation)
                        social_features['XY_FEATURES'] = all_features[:social_features['XY_FEATURES'].shape[0], :]
                        if 'LABELS' in social_features:
                            social_features['LABELS'] = all_features[social_features['XY_FEATURES'].shape[0]:, :]

                    # Get xy coordinates
                    padded_xy = np.zeros((self.timesteps, 2), dtype=np.float)
                    padded_xy[input_mask] = social_features['XY_FEATURES']
                    social['social_xy_features'].append(padded_xy)

                    # Get labels
                    labels = np.array([])
                    if 'LABELS' in social_features:
                        labels = social_features['LABELS']
                    padded_labels = np.zeros((self.outsteps, 2), dtype=np.float)
                    if len(labels) > 0:
                        padded_labels[label_mask] = labels
                    social['social_xy_labels'].append(padded_labels)

                    if len(labels) == 0 or self.mode != 'train':
                        social_str = "_FULL" if self.is_oracle else "_PARTIAL"
                    else:
                        social_str = self.str
                    # Get centerline for IFC

                    if self.mode == 'train':
                        social_features[self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+social_str] = self.add_noise(social_features[self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+social_str], rotation=rotation, translation=translation)
                    social['social_oracle_centerline'].append(social_features[self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+social_str])
                    social['social_oracle_centerline_lengths'].append(social_features[self.heuristic_str + 'ORACLE_CENTERLINE_NORMALIZED'+social_str].shape[0])              
                    num_social_agents += 1

        # Pad centerlines
        social_max_pad = np.max(social['social_oracle_centerline_lengths'])
        if social_max_pad < np.max(example['agent_oracle_centerline_lengths']):
            social_max_pad = np.max(example['agent_oracle_centerline_lengths'])
        for index, elem in enumerate(social['social_oracle_centerline']):
            num_pad = social_max_pad - elem.shape[0]
            padded_elem = np.pad(elem, ((0, num_pad), (0, 0)), 'constant', constant_values=(0.,))
            social['social_oracle_centerline'][index] = padded_elem

        if self.is_oracle:
            max_pad = social_max_pad
            for index, elem in enumerate(example['agent_oracle_centerline']):
                num_pad = max_pad - elem.shape[0]
                padded_elem = np.pad(elem, ((0, num_pad),(0, 0)), 'constant', constant_values=(0.,))
                example['agent_oracle_centerline'][index] = padded_elem
            example['agent_oracle_centerline'] = np.array(example['agent_oracle_centerline'])
            example['agent_oracle_centerline_lengths'] = np.array(example['agent_oracle_centerline_lengths'])
        else:
            example['agent_oracle_centerline'] = np.pad(example['agent_oracle_centerline'], ((0, social_max_pad - example['agent_oracle_centerline'].shape[0]),(0,0)), 'constant', constant_values=(0.,))

        social = {key: np.array(value) for key, value in social.items()}

        # Compute delta xy coordinates if required
        if self.delta:
            padded_social_xy_delta, padded_social_labels_delta, social_ref_start, social_ref_end = self.relative_distance_with_labels(social['social_xy_features'], social['social_xy_labels'])
            social['social_xy_features_delta'] = padded_social_xy_delta
            social['social_xy_labels_delta'] = padded_social_labels_delta
            social['social_xy_ref_start'] = social_ref_start
            social['social_xy_ref_end'] = social_ref_end

        example.update(social)
        example['num_social_agents'] = num_social_agents

        # Create adjacency matrix
        adjacency = np.zeros((self.timesteps, num_social_agents+1, num_social_agents+1))
        label_adjacency = np.zeros((self.outsteps, num_social_agents+1, num_social_agents+1))

        # Focal agent is always present
        # Remove self loop
        adjacency[:, 0, :] = 1
        label_adjacency[:, 0, :] = 1
        for social_agent, input_mask in enumerate(example['social_input_mask']):
            adjacency[input_mask, social_agent + 1, :] = 1
        for social_agent, input_mask in enumerate(example['social_label_mask']):
            label_adjacency[input_mask, social_agent + 1, :] = 1
        indexer = np.arange(num_social_agents + 1)
        adjacency[:, indexer, indexer] = 0
        label_adjacency[:, indexer, indexer] = 0

        example['adjacency'] = adjacency
        example['label_adjacency'] = label_adjacency

        '''
        get_data_from_batch
        '''
        # Get focal agent features
        agent_features = example['agent_xy_features' + self.delta_str]
        if self.map_features_flag:
            agent_features = torch.cat([agent_features, example['agent_map_features']], dim=-1)
        agent_features = agent_features.astype(np.float32)

        # Get social features
        social_features = example['social_xy_features' + self.delta_str]
        social_label_features = example['social_xy_labels' + self.delta_str]
        social_features = social_features.astype(np.float32)
        social_label_features = social_label_features.astype(np.float32)
        social_input_mask = example['social_input_mask']
        social_label_mask = example['social_label_mask']
        num_agent_mask = np.ones(example['num_social_agents'] + 1, dtype=np.float32)
        # num_agent_mask = (example['num_social_agents'][:, None] >= torch.arange(social_label_mask.size(1) + 1)).astype(np.float32)
        adjacency = example['adjacency'].astype(np.float32)
        label_adjacency = example['label_adjacency'].astype(np.float32)

        # Get labels
        agent_labels = example['agent_xy_labels' + self.delta_str].astype(np.float32)
        social_labels = example['social_xy_labels' + self.delta_str].astype(np.float32)

        # Get IFC features
        ifc_helpers = {}
        ifc_helpers['agent_oracle_centerline'] = example['agent_oracle_centerline'].astype(np.float32)
        ifc_helpers['agent_oracle_centerline_lengths'] = np.int64(example['agent_oracle_centerline_lengths'])
        # ifc_helpers['agent_xy_delta'] = None

        ifc_helpers['social_oracle_centerline'] = example['social_oracle_centerline'].astype(np.float32)
        ifc_helpers['social_oracle_centerline_lengths'] = np.int64(example['social_oracle_centerline_lengths'])
        # ifc_helpers['social_xy_delta'] = None

        ifc_helpers['rotation'] = example['rotation']
        ifc_helpers['translation'] = example['translation']
        ifc_helpers['city'] = example['city']
        ifc_helpers['idx'] = example['seq_id']

        if self.delta:
            ifc_helpers['agent_xy_delta'] = example['agent_xy_ref_end'].astype(np.float32)
            ifc_helpers['social_xy_delta'] = example['social_xy_ref_end'].astype(np.float32)

        input_dict = {'agent_features': agent_features,
                      'ifc_helpers': ifc_helpers,
                      'social_features': social_features,
                      'social_label_features': social_label_features,
                      'adjacency': adjacency,
                      'label_adjacency': label_adjacency,
                      'num_agent_mask': num_agent_mask
                      }

        if self.mode != 'test':
            target_dict = {'agent_labels': agent_labels,
                        #    'agent_xy_ref_end': ifc_helpers['agent_xy_delta'] if self.delta else None,
                        #    'social_labels': social_labels,
                        #    'social_label_mask': social_label_mask.astype(np.float32),
                        #    'idx': example['seq_id']
                           }
            return input_dict, target_dict
        else:
            return input_dict, None

    def denormalize_xy(self, xy_locations, translation=None, rotation=None):
        """Reverse the Translate and rotate operations on the input data
            Args:
                xy_locations (numpy array): XY positions for the trajectory
            Returns:
                xy_locations_normalized (numpy array): denormalized XY positions
        """
        # Apply rotation
        num = xy_locations.shape[0]
        if xy_locations.shape[0] > 1:
            trajectory = LineString(xy_locations)
        else:
            trajectory = LineString(np.concatenate(([[0.0, 0.0]], xy_locations), axis=0))

        if rotation is not None:
            trajectory = rotate(trajectory, rotation, origin=(0, 0))

        if translation is not None:
            mat = [1, 0, 0, 1, translation[0], translation[1]]
            trajectory = affine_transform(trajectory, mat)

        output = np.array(trajectory.coords, dtype=np.float32)
        if num <= 1:
            output = output[1:]

        return output

    def add_noise(self, x, rotation, translation):
        trajectory = LineString(x)
        mat = [1, 0, 0, 1, translation[0], translation[1]]
        trajectory_translated = affine_transform(trajectory, mat)

        # Apply rotation
        trajectory_rotated = np.array(rotate(trajectory_translated, rotation, origin=(0, 0)).coords, dtype=np.float32)
        return trajectory_rotated

    def relative_distance_with_labels(self, input, labels):
        """Compute relative distance from absolute
            Returns:
                reference: First element of the trajectory. Enables going back from relative distance to absolute.
        """
        if len(input.shape) == 3:
            # Change input sequences to relative distances
            input_reference_start = input[:, 0, :]
            input_reference_end = input[:, -1, :]
            input_rel_dist = input - np.pad(input, ((0, 0), (1, 0), (0, 0)), 'constant')[:, :input.shape[1], :]

            # Change output sequences to relative distances
            output_rel_dist = labels - np.concatenate((input[:, -1:, :], labels), axis=1)[:, :labels.shape[1], :]
        else:
            # Change input sequences to relative distances
            input_reference_start = input[0, :]
            input_reference_end = input[-1, :]
            input_rel_dist = input - np.pad(input, ((1, 0), (0, 0)), 'constant')[:input.shape[0], :]

            # Change output sequences to relative distances
            output_rel_dist = labels - np.concatenate((input[-1:,:], labels), axis=0)[:labels.shape[0], :]

        return input_rel_dist, output_rel_dist, input_reference_start, input_reference_end

    @staticmethod
    def collate(batch):
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
        numpy_type_map = {
            'float64': torch.DoubleTensor,
            'float32': torch.FloatTensor,
            'float16': torch.HalfTensor,
            'int64': torch.LongTensor,
            'int32': torch.IntTensor,
            'int16': torch.ShortTensor,
            'int8': torch.CharTensor,
            'uint8': torch.ByteTensor,
        }

        def pad_batch(batch_dict, max_actors):
            '''
                Pad batch such that all examples have same number of social actors. Allows for batch training of graph models.
            '''
            for key, value in batch_dict.items():
                if key == 'social_oracle_centerline':
                    max_centerline_pad = np.max([x.size(1) for x in value])
                if isinstance(value, dict):
                    batch_dict[key] = pad_batch(value, max_actors)
                elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    if 'agent' not in key:
                        for index, elem in enumerate(value):
                            if 'adjacency' not in key:
                                if 'centerline' in key and 'lengths' not in key:
                                    num_centerline_pad = max_centerline_pad - elem.size(1)
                                    if len(elem.size()) == 3:
                                        elem = torch.nn.functional.pad(elem, (0, 0, 0, num_centerline_pad, 0, 0), value=0.)
                                    else:
                                        elem = torch.nn.functional.pad(elem, (0, num_centerline_pad, 0, 0), value=0.)
                                num_pad = max_actors - elem.size(0)
                                if len(elem.size()) == 3:
                                    padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_pad))
                                elif len(elem.size()) == 2:
                                    padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, num_pad))
                                else:
                                    padded_elem = torch.nn.functional.pad(elem, (0, num_pad))
                            else:
                                num_pad = max_actors - elem.size(1) + 1
                                padded_elem = torch.nn.functional.pad(elem, (0, num_pad, 0, num_pad, 0, 0))
                            value[index] = padded_elem
                        batch_dict[key] = torch.stack(value)
                    else:
                        try:
                            if ('centerline' in key and 'lengths' not in key) or ('mask' in key):
                                max_pad = np.max([x.size(0) for x in value])
                                for index, elem in enumerate(value):
                                    num_pad = max_pad - elem.size(0)
                                    if len(elem.size()) == 3:
                                        padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_pad), value=0.)
                                    elif len(elem.size()) == 1:
                                        padded_elem = torch.nn.functional.pad(elem, (0, num_pad), value=0.)
                                    else:
                                        padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, num_pad), value=0.)
                                    value[index] = padded_elem
                            batch_dict[key] = torch.stack(value)
                        except:
                            if 'centerline' in key and 'lengths' not in key:
                                max_pad = np.max([x.size(1) for x in value])
                                for index, elem in enumerate(value):
                                    num_pad = max_pad - elem.size(1)
                                    if len(elem.size()) == 3:
                                        padded_elem = torch.nn.functional.pad(elem, (0,0,0,num_pad,0,0), value=0.)
                                    else:
                                        padded_elem = torch.nn.functional.pad(elem, (0,0,0,num_pad), value=0.)
                                    value[index] = padded_elem
                            max_actors = 6
                            for index, elem in enumerate(value):
                                num_pad = max_actors - elem.size(0)
                                if len(elem.size()) == 3:
                                    padded_elem = torch.nn.functional.pad(elem, (0,0,0,0,0,num_pad))
                                elif len(elem.size()) == 2:
                                    padded_elem = torch.nn.functional.pad(elem, (0,0,0,num_pad))
                                else:
                                    padded_elem = torch.nn.functional.pad(elem, (0,num_pad))
                                value[index] = padded_elem
                            batch_dict[key] = torch.stack(value)
            return batch_dict

        def collate_batch(batch):
            """Puts each data field into a tensor with outer dimension batch size"""
            elem_type = type(batch[0])
            if isinstance(batch[0], torch.Tensor):
                out = None
                if False:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = batch[0].storage()._new_shared(numel)
                    out = batch[0].new(storage)
                try:
                    return torch.stack(batch, 0, out=out)
                except:
                    return batch
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                elem = batch[0]
                if elem_type.__name__ == 'ndarray':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(error_msg_fmt.format(elem.dtype))

                    return collate_batch([torch.from_numpy(b) for b in batch])
                if elem.shape == ():  # scalars
                    py_type = float if elem.dtype.name.startswith('float') else int
                    return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
            elif isinstance(batch[0], float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(batch[0], int_classes):
                return torch.tensor(batch)
            elif isinstance(batch[0], string_classes):
                return batch
            elif isinstance(batch[0], container_abcs.Mapping):
                return {key: collate_batch([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
                return type(batch[0])(*(collate_batch(samples) for samples in zip(*batch)))
            elif isinstance(batch[0], container_abcs.Sequence):
                transposed = zip_longest(*batch)
                return [collate_batch(samples) for samples in transposed]
            else:
                return batch
            raise TypeError((error_msg_fmt.format(type(batch[0]))))

        batch = collate_batch(batch)
        max_actors = np.max([x.shape[0] for x in batch[0]['social_features']])
        batch[0] = pad_batch(batch[0], max_actors)
        batch[1] = pad_batch(batch[1], max_actors)
        return batch
