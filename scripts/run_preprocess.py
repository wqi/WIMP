import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import parmap
import pickle as pkl
import warnings
from scenario_serialization import load_argoverse_scenario_hdf5, load_static_map_json
import sys
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    remove_overlapping_lane_seq,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from collections import OrderedDict
from map_features_utils import MapFeaturesUtils
from tqdm import tqdm
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union # NOQA
from argoverse.utils.manhattan_search import (
    compute_polygon_bboxes,
    find_all_polygon_bboxes_overlapping_query_bbox,
    find_local_polygons,
)
from collections import defaultdict
from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    lane_waypt_to_query_dist,
    remove_overlapping_lane_seq,
)
warnings.simplefilter('ignore', np.RankWarning)


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, required=True, help="Path to .csv data files")
    parser.add_argument('--mode', type=str, required=True,
                        help="Data to be processed - train|val|test")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="Folder where processed data should be saved")
    parser.add_argument('--social-features', action='store_true', help="Compute social features")
    parser.add_argument('--map-features', action='store_true', help="Compute map features")
    parser.add_argument('--xy-features', action='store_true', help="Compute XY features")
    parser.add_argument('--normalize', action='store_true', help="Normalize XY features")
    parser.add_argument('--ego-features', action='store_true',
                        help="Compute ego-centric XY features")
    parser.add_argument('--extra-map-features', action='store_true',
                        help="Compute curvature and nearest neighbour centerline features")
    parser.add_argument('--centerline-span', type=int, default=10,
                        help="Number of points on either side of closes centerline point. \
                              Only applicable with --extra-map-features flag.")
    parser.add_argument('--timesteps', type=int, default=20, help="Length of input timesteps")
    parser.add_argument('--test-labels-path', type=str, default="",
                        help="Path to test labels (if available)")
    parser.add_argument('--generate-candidate-centerlines', type=int, default=0,
                        help='Generate candidate centerlines using heuristic')
    parser.add_argument('--compute-all', action='store_true', help='Compute all map features')
    parser.add_argument('--start-file', type=int, default=0, help='Start file index')
    parser.add_argument('--chunk-length', type=int, default=-1, help='Chunk length')
    parser.add_argument('--num-cpus', type=int, default=os.cpu_count(),
                        help='Number of cpus to use')
    args = parser.parse_args()
    return args


class ModifiedArgoverseMap(ArgoverseMap):
    def __init__(self):
        super(ModifiedArgoverseMap, self).__init__()

    def get_candidate_centerlines_for_traj(self, xy: np.ndarray, city_name: str, viz: bool = False,
                                           max_search_radius: float = 50.0,
                                           manhattan_threshold: float = 2.5,
                                           dfs_threshold_multiplier: float = 2.0) \
            -> List[np.ndarray]:
        """
        Copied from map_api.py. Made changes to traverse the graph further
        Get centerline candidates upto a threshold.

        Algorithm:
        1. Take the lanes in the bubble of last obs coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines with max distance along centerline

        Args:
            xy: trajectory of shape (N, 2).
            city_name
            viz: Visualize

        Returns:
            candidate_centerlines: List of candidate centerlines
        """

        # Get all lane candidates within a bubble
        curr_lane_candidates = self.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name,
                                                            manhattan_threshold)

        # Keep expanding the bubble until at least 1 lane is found
        while len(curr_lane_candidates) < 1 and manhattan_threshold < max_search_radius:
            manhattan_threshold *= 2
            curr_lane_candidates = self.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name,
                                                                manhattan_threshold)

        assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

        # Set dfs threshold
        displacement = np.sqrt((xy[0, 0] - xy[-1, 0]) ** 2 + (xy[0, 1] - xy[-1, 1]) ** 2)
        dfs_threshold = displacement * dfs_threshold_multiplier

        # DFS to get all successor and predece ssor candidates
        obs_pred_lanes: List[Sequence[int]] = []
        for lane in curr_lane_candidates:
            candidates_future = self.dfs(lane, city_name, 0, dfs_threshold)
            candidates_past = self.dfs(lane, city_name, 0, dfs_threshold, True)

            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate \
                                                                     lanes past and future"
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Remove unnecessary extended predecessors
        obs_pred_lanes = self.remove_extended_predecessors(obs_pred_lanes, xy, city_name)

        # Getting candidate centerlines
        candidate_cl = self.get_cl_from_lane_seq(obs_pred_lanes, city_name)

        # Reduce the number of candidates based on distance travelled along the centerline
        candidate_centerlines = filter_candidate_centerlines(xy, candidate_cl)

        # If no candidate found using above criteria, take the onces along with travel is the maximum # NOQA
        if len(candidate_centerlines) < 1:
            candidate_centerlines = get_centerlines_most_aligned_with_trajectory(xy, candidate_cl)

        if viz:
            plt.figure(0, figsize=(8, 7))
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            plt.plot(xy[:, 0], xy[:, 1], "-", color="#d33e4c", alpha=1, linewidth=1, zorder=15)

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            plt.plot(final_x, final_y, "o", color="#d33e4c", alpha=1, markersize=7, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("Number of candidates = {}".format(len(candidate_centerlines)))
            plt.show()

        return candidate_centerlines


class ScenarioMap:
    def __init__(self, root):
        """Initialize the Argoverse Map."""
        self.root = root
        self.render_window_radius = 150
        self.im_scale_factor = 50
        self.data = load_static_map_json(root)
        self.city_lane_centerlines_dict, self.predecessors_dict, self.successors_dict = self.build_centerline_index()
        (
            self.city_halluc_bbox_table,
            self.city_halluc_tableidx_to_laneid_map,
        ) = self.build_hallucinated_lane_bbox_index()

        # get hallucinated lane extends and driveable area from binary img
        self.city_to_lane_polygons_dict: Mapping[str, np.ndarray] = {}
        self.city_to_driveable_areas_dict: Mapping[str, np.ndarray] = {}
        self.city_to_lane_bboxes_dict: Mapping[str, np.ndarray] = {}
        self.city_to_da_bboxes_dict: Mapping[str, np.ndarray] = {}

        # for city_name in self.city_name_to_city_id_dict.keys():
        #     lane_polygons = np.array(self.get_vector_map_lane_polygons(city_name), dtype=object)
        #     driveable_areas = np.array(self.get_vector_map_driveable_areas(city_name), dtype=object)
        #     lane_bboxes = compute_polygon_bboxes(lane_polygons)
        #     da_bboxes = compute_polygon_bboxes(driveable_areas)

        #     self.city_to_lane_polygons_dict[city_name] = lane_polygons
        #     self.city_to_driveable_areas_dict[city_name] = driveable_areas
        #     self.city_to_lane_bboxes_dict[city_name] = lane_bboxes
        #     self.city_to_da_bboxes_dict[city_name] = da_bboxes

    def build_centerline_index(self):
        """
        Build dictionary of centerline for each city, with lane_id as key
        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are dictionaries
                                        (k=lane_id, v=lane info)
        """
        city_lane_centerlines_dict = {}
        predecessors_dict = defaultdict(list)
        successors_dict = defaultdict(list)
        for lane in self.data['lane_segments']:
            city_lane_centerlines_dict[lane['id']] = lane
            for x in lane['successors']:
                successors_dict[lane['id']].append(x)
                predecessors_dict[x].append(lane['id'])
        return city_lane_centerlines_dict, predecessors_dict, successors_dict
    
    def build_hallucinated_lane_bbox_index(self):
        """
        Populate the pre-computed hallucinated extent of each lane polygon, to allow for fast
        queries.
        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = []
        city_halluc_tableidx_to_laneid_map = {}

        for lane in self.data['lane_segments']:
            left_lane_xy_start = np.array([lane['left_lane_boundary']['points'][0]['x'], lane['left_lane_boundary']['points'][0]['y']])
            left_lane_xy_end = np.array([lane['left_lane_boundary']['points'][-1]['x'], lane['left_lane_boundary']['points'][-1]['y']])
            right_lane_xy_start = np.array([lane['right_lane_boundary']['points'][0]['x'], lane['right_lane_boundary']['points'][0]['y']])
            right_lane_xy_end = np.array([lane['right_lane_boundary']['points'][-1]['x'], lane['right_lane_boundary']['points'][-1]['y']])
            area_1 = np.abs(left_lane_xy_start[0] - right_lane_xy_end[0]) * np.abs(left_lane_xy_start[1] - right_lane_xy_end[1])
            area_2 = np.abs(right_lane_xy_start[0] - left_lane_xy_end[0]) * np.abs(right_lane_xy_start[1] - left_lane_xy_end[1])
            city_halluc_tableidx_to_laneid_map[str(len(city_halluc_bbox_table))] = lane['id']
            if area_1 > area_2:
                city_halluc_bbox_table.append([left_lane_xy_start[0], left_lane_xy_start[1], right_lane_xy_end[0], right_lane_xy_end[1]])
            else:
                city_halluc_bbox_table.append([right_lane_xy_start[0], right_lane_xy_start[1], left_lane_xy_end[0], left_lane_xy_end[1]])
        city_halluc_bbox_table = np.array(city_halluc_bbox_table)  
        return city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map

    def get_lane_ids_in_xy_bbox(
        self,
        query_x,
        query_y,
        query_search_range_manhattan = 5.0,
    ):
        """
        Prune away all lane segments based on Manhattan distance. We vectorize this instead
        of using a for-loop. Get all lane IDs within a bounding box in the xy plane.
        This is a approximation of a bubble search for point-to-polygon distance.
        The bounding boxes of small point clouds (lane centerline waypoints) are precomputed in the map.
        We then can perform an efficient search based on manhattan distance search radius from a
        given 2D query point.
        We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
        hallucinated lane polygon extents.
        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            query_search_range_manhattan: search radius along axes
        Returns:
            lane_ids: lane segment IDs that live within a bubble
        """
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            self.city_halluc_bbox_table,
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        if len(overlap_indxs) == 0:
            return []

        neighborhood_lane_ids: List[int] = []
        for overlap_idx in overlap_indxs:
            lane_segment_id = self.city_halluc_tableidx_to_laneid_map[str(overlap_idx)]
            neighborhood_lane_ids.append(lane_segment_id)

        return neighborhood_lane_ids

    def get_lane_segment_predecessor_ids(self, lane_segment_id):
        """
        Get land id for the lane predecessor of the specified lane_segment_id
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
            predecessor_ids: list of integers, representing lane segment IDs of predecessors
        """
        predecessor_ids = self.predecessors_dict[lane_segment_id]
        return predecessor_ids

    def get_lane_segment_successor_ids(self, lane_segment_id):
        """
        Get land id for the lane sucessor of the specified lane_segment_id
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh
        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors
        """
        successor_ids = self.successors_dict[lane_segment_id]
        return successor_ids

    def get_lane_segment_centerline(self, lane_segment_id):
        """
        We return a 3D centerline for any particular lane segment.
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh
        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """
        try:
            lane_centerline = np.array([[centerline['x'], centerline['y'], centerline['z']] for centerline in self.city_lane_centerlines_dict[lane_segment_id]['centerline']['points']])
            return lane_centerline
        except:
            return None
        
    def get_lane_segment_polygon(self, lane_segment_id: int) -> np.ndarray:
        """
        Hallucinate a 3d lane polygon based around the centerline. We rely on the average
        lane width within our cities to hallucinate the boundaries. We rely upon the
        rasterized maps to provide heights to points in the xy plane.
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh
        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        lane_centerline = self.get_lane_segment_centerline(lane_segment_id)
        try:
            lane_polygon = centerline_to_polygon(lane_centerline[:, :2])
        except:
            import ipdb; ipdb.set_trace()
            a = 1
        return np.hstack([lane_polygon, np.zeros(lane_polygon.shape[0])[:, np.newaxis] + np.mean(lane_centerline[:, 2])])

    def get_cl_from_lane_seq(self, lane_seqs: Iterable[List[int]]) -> List[np.ndarray]:
        """Get centerlines corresponding to each lane sequence in lane_sequences
        Args:
            lane_seqs: Iterable of sequence of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
            candidate_cl: list of numpy arrays for centerline corresponding to each lane sequence
        """

        candidate_cl = []
        for lanes in lane_seqs:
            curr_candidate_cl = np.empty((0, 2))
            for curr_lane in lanes:
                curr_candidate = self.get_lane_segment_centerline(curr_lane)[:, :2]
                curr_candidate_cl = np.vstack((curr_candidate_cl, curr_candidate))
            candidate_cl.append(curr_candidate_cl)
        return candidate_cl

    def dfs(
        self,
        lane_id: int,
        dist: float = 0,
        threshold: float = 30,
        extend_along_predecessor: bool = False,
    ) -> List[List[int]]:
        """
        Perform depth first search over lane graph up to the threshold.
        Args:
            lane_id: Starting lane_id (Eg. 12345)
            city_name
            dist: Distance of the current path
            threshold: Threshold after which to stop the search
            extend_along_predecessor: if true, dfs over predecessors, else successors
        Returns:
            lanes_to_return (list of list of integers): List of sequence of lane ids
                Eg. [[12345, 12346, 12347], [12345, 12348]]
        """
        if dist > threshold:
            return [[lane_id]]
        else:
            traversed_lanes = []
            child_lanes = (
                self.get_lane_segment_predecessor_ids(lane_id)
                if extend_along_predecessor
                else self.get_lane_segment_successor_ids(lane_id)
            )
            if child_lanes is not None:
                for child in child_lanes:
                    centerline = self.get_lane_segment_centerline(child)
                    if centerline is not None:
                        try:
                            cl_length = LineString(centerline).length
                            curr_lane_ids = self.dfs(
                                child,
                                dist + cl_length,
                                threshold,
                                extend_along_predecessor,
                            )
                            traversed_lanes.extend(curr_lane_ids)
                        except:
                            pass

            if len(traversed_lanes) == 0:
                return [[lane_id]]
            lanes_to_return = []
            for lane_seq in traversed_lanes:
                lanes_to_return.append(lane_seq + [lane_id] if extend_along_predecessor else [lane_id] + lane_seq)
            return lanes_to_return


def compute_features(path, xy_features_flag=True, xy_features_normalize_flag=True,
                     map_features_flag=True, social_features_flag=True, timesteps=20, avm=None,
                     mfu=None, return_labels=False, label_path="", generate_candidate_centerlines=0,
                     compute_all=False):
    """
    Compute features for the given sequence
        Args:
            path (str): Location of the .csv sequence data file
            xy_features_flag (bool): Use xy coordinate features
            xy_features_normalize_flag (bool): Normalize xy features to constraint start of
                sequence to be (0,0) and end of sequence to be on the positive x axis
            map_features_flag (bool): Compute map features if true
            social_features_flag (bool): Compute social features if true
            timesteps (int): Number of input timesteps (10 timesteps = 1 second)
            avm (ArgoverseMap object): default None. Pass an object if calling this function in a
                loop to avoid redundant computation
            return_labels (bool): Compute the labels for the given sequence if true
            label_path (str): Path to separate label data file, if necessary

        Returns:
            features (dict): Dictionary of features
            feature_helpers (dict) (Only when map features are computed): Dictionary that stores
                statistics of transformations for map features
    """

    def normalize_xy(xy_locations, translation=None, rotation=None, to_rotate=True):
        """
        Translate and rotate the input data so that the first timestep is (0,0) and the last
        timestep lies on the positive x axis.
            Args:
                xy_locations (numpy array): XY positions for the trajectory
            Returns:
                xy_locations_normalized (numpy array): normalized XY positions
                feature_helpers (dict): Dictionary that stores the rotations and translations
                    applied to the trajectory
        """
        # Apply translation
        if xy_locations.shape[0] > 1:
            trajectory = LineString(xy_locations)
            if translation is None:
                translation = [-xy_locations[0, 0], -xy_locations[0, 1]]
            mat = [1, 0, 0, 1, translation[0], translation[1]]
            trajectory_translated = affine_transform(trajectory, mat)

            # Apply rotation
            if to_rotate:
                if rotation is None:
                    rotation = -np.degrees(np.arctan2(trajectory_translated.coords[-1][1],
                                           trajectory_translated.coords[-1][0]))

                trajectory_rotated = np.array(rotate(trajectory_translated, rotation,
                                              origin=(0, 0)).coords)
                return trajectory_rotated, {'TRANSLATION': translation, 'ROTATION': rotation}
            else:
                return trajectory_translated, {'TRANSLATION': translation, 'ROTATION': None}
        else:
            if translation is None:
                return np.zeros_like(xy_locations, dtype=np.float), \
                       {'TRANSLATION': [-xy_locations[0, 0], -xy_locations[0, 1]], 'ROTATION': None}
            else:
                return np.array([[xy_locations[0, 0]+translation[0],
                                 xy_locations[0, 0]+translation[1]]]), \
                       {'TRANSLATION': translation, 'ROTATION': None}

    def compute_xy_features(xy_locations, normalize=True, timesteps=20):
        """
        Compute XY features for the given sequence
            Args:
                xy_locations (numpy array): XY positions for the track
                normalize (bool): Normalize xy features to constraint start of sequence to be (0,0)
                    and end of sequence to be on the positive x axis
                timesteps (int): Timesteps for which feature computation needs to be done
                    (10 timesteps = 1 second)
            Returns:
                xy_features (numpy array): XY features for the given input positions
                feature_helpers (dict) (Only when normalize=True): Translation and rotations
                    applied to the input data. This information can be used later to denormalize
                    the features.
        """
        # Apply normalization
        if normalize:
            xy_locations, feature_helpers = normalize_xy(xy_locations)
            return xy_locations, feature_helpers
        return xy_locations, None

    def compute_map_features(xy_locations, map_json, timesteps=20, avm=None, mfu=None, rotation=None,
                             translation=None, labels=None, generate_candidate_centerlines=0,
                             compute_all=False):
        """
        Compute map based features for the given sequence
            Args:
                xy_locations (numpy array): XY positions for the track
                city (string): Name of the city
                timesteps (int): Timesteps for which feature computation needs to be done
                    (10 timesteps = 1 second)
                avm (ArgoverseMap object): default None. Pass an object if calling this function in
                    a loop to avoid redundant computation
            Returns:
                nt_distances_oracle (numpy array): normal and tangential distances for oracle
                    centerline
                map_feature_helpers (dict): Dictionary containing helpers for map features
        """
        def remove_repeated(centerlines):
            remove_elements = np.zeros(len(centerlines))
            for x in range(len(centerlines)):
                for y in range(x+1, len(centerlines)):
                    if centerlines[x].shape == centerlines[y].shape:
                        if remove_elements[y] == 0:
                            if np.all(centerlines[x] == centerlines[y]):
                                remove_elements[y] = 1
            return np.array(centerlines)[remove_elements == 0]

        def additional_centerline_features(centerline, xy_locations, save_str="", heuristic=False):
            """
            Compute additional centerline features like curvature and nearest neighbours to
            xy location
            """
            # Compute nearest point to each xy location on the centerline
            distances = cdist(xy_locations, centerline)
            min_dist = np.argmin(distances, axis=1)
            indexing_array = []
            heuristic_str = 'HEURISTIC_' if heuristic else ''
            for i in range(-args.centerline_span, args.centerline_span):
                loc = min_dist + i
                min_mask = loc < 0
                max_mask = loc >= centerline.shape[0]

                loc[min_mask] = 0
                loc[max_mask] = centerline.shape[0] - 1
                indexing_array.append(loc)

            indexing_array = np.stack(indexing_array, axis=1)
            arange_array = np.arange(indexing_array.shape[0])
            centerline_features = np.repeat(np.expand_dims(centerline, axis=0),
                                            indexing_array.shape[0], axis=0)[arange_array[:, None],
                                                                             indexing_array, :]

            # Compute angle of change
            velocity_vector = (centerline_features - np.pad(centerline_features, ((0, 0), (1, 0), (0, 0)), 'constant')[:, :centerline_features.shape[1], :])[:, 1:, :] # NOQA
            absolute_angles = np.degrees(np.arctan2(velocity_vector[:, :, 1],
                                         velocity_vector[:, :, 0]))
            relative_angles = (absolute_angles - np.pad(absolute_angles, ((0, 0), (1, 0)), 'constant')[:, :absolute_angles.shape[1]])[:, 1:] # NOQA

            # Compute angle between centerline and heading
            heading_vector = (xy_locations - np.pad(xy_locations, ((1, 0), (0, 0)), 'constant')[:xy_locations.shape[0], :]) # NOQA
            heading_angle = np.degrees(np.arctan2(heading_vector[:, 1], heading_vector[:, 0]))
            relative_heading_angle = np.abs(absolute_angles - heading_angle[:, None])

            features = {
                    heuristic_str + 'NEAREST_CENTERLINE_FEATURES' + save_str: centerline_features,
                    heuristic_str + 'NEAREST_VELOCITY_VECTOR' + save_str: velocity_vector,
                    heuristic_str + 'NEAREST_ABSOLUTE_ANGLE' + save_str: absolute_angles,
                    heuristic_str + 'NEAREST_RELATIVE_ANGLE' + save_str: relative_angles,
                    heuristic_str + 'NEAREST_RELATIVE_HEADING_ANGLE' + save_str: relative_heading_angle, # NOQA
                    heuristic_str + 'NEAREST_HEADING_ANGLE' + save_str: heading_angle}

            # Compute ego centric features
            if args.ego_features:
                translation = -xy_locations
                rotation = -np.degrees(np.arctan2(heading_vector[:, 1], heading_vector[:, 0]))
                for key in [heuristic_str + 'NEAREST_CENTERLINE_FEATURES' + save_str]:
                    ego_feature = []
                    for index, point in enumerate(xy_locations):
                        ego_feature.append(normalize_xy(features[key][index],
                                           translation=translation[index],
                                           rotation=rotation[index])[0])
                    features[heuristic_str + 'EGO_'+key] = np.array(ego_feature)
            return features

        def map_features_helper(locations, dfs_threshold_multiplier=2.0, save_str="", avm=None,
                                mfu=None, rotation=None, translation=None,
                                generate_candidate_centerlines=0, compute_all=False):
            # Initialize map utilities if not provided
            if avm is None:
                avm = ArgoverseMap()
            if mfu is None:
                mfu = MapFeaturesUtils()
            
            # Get best-fitting (oracle) centerline for current vehicle
            heuristic_oracle_centerline = mfu.get_candidate_centerlines_for_trajectory(locations, map_json, avm=avm, viz=False, max_candidates=generate_candidate_centerlines, mode='train')[0] # NOQA
            features = {
                "HEURISTIC_ORACLE_CENTERLINE" + save_str: heuristic_oracle_centerline,
                "HEURISTIC_ORACLE_CENTERLINE_NORMALIZED" + save_str: normalize_xy(heuristic_oracle_centerline, translation=translation, rotation=rotation)[0] # NOQA
            }

            # Get top-fitting candidate centerlines for current vehicle (can beused at test time)
            if compute_all:
                if generate_candidate_centerlines > 0:
                    test_candidate_centerlines = mfu.get_candidate_centerlines_for_trajectory(locations, map_json, avm=avm, viz=False, max_candidates=generate_candidate_centerlines, mode='test') # NOQA
                    features["TEST_CANDIDATE_CENTERLINES" + save_str] = test_candidate_centerlines

                # Apply rotation and translation normalization if specified
                if rotation is not None or translation is not None:
                    if generate_candidate_centerlines > 0:
                        features['TEST_CANDIDATE_CENTERLINE_NORMALIZED' + save_str] = [normalize_xy(test_candidate_centerline, translation=translation, rotation=rotation)[0] for test_candidate_centerline in test_candidate_centerlines] # NOQA
            return features

        map_features = {}

        # Compute polyline-based map features considering only first 2 seconds
        xy_partial_trajectory = xy_locations[:timesteps, :]
        map_partial_features = map_features_helper(xy_partial_trajectory, save_str="_PARTIAL",
                                                   avm=avm, rotation=rotation,
                                                   translation=translation,
                                                   generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                   compute_all=compute_all)
        map_features.update(map_partial_features)

        # Compute polyline-based map features considering 5 seconds
        if labels is not None:
            xy_full_trajectory = np.concatenate([xy_locations[:timesteps, :], labels], axis=0)
            map_full_features = map_features_helper(xy_full_trajectory, save_str="_FULL", avm=avm,
                                                    rotation=rotation, translation=translation,
                                                    generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                    compute_all=compute_all)
            map_features.update(map_full_features)
        
        # Compute extra map features if specified
        if args.extra_map_features:
            rotated_and_translated_partial_trajectory = normalize_xy(xy_partial_trajectory,
                                                                     translation=translation,
                                                                     rotation=rotation)[0]
            if labels is not None:
                if len(labels) > 1:
                    rotated_and_translated_label = normalize_xy(labels, translation=translation,
                                                                rotation=rotation)[0]
                else:
                    rotated_and_translated_label = normalize_xy(labels, translation=translation,
                                                                rotation=None, to_rotate=False)[0]

            heuristic_extra_features_partial = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"], rotated_and_translated_partial_trajectory, save_str="_PARTIAL", heuristic=True) # NOQA
            map_features.update(heuristic_extra_features_partial)
            if labels is not None:
                heuristic_extra_features_full = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_FULL"], rotated_and_translated_partial_trajectory, save_str="_FULL", heuristic=True) # NOQA
                heuristic_extra_features_label_full = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_FULL"], rotated_and_translated_label, save_str="_LABEL_FULL", heuristic=True) # NOQA
                heuristic_extra_features_label_partial = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"], rotated_and_translated_label, save_str="_LABEL_PARTIAL", heuristic=True) # NOQA
                map_features.update(heuristic_extra_features_full)
                map_features.update(heuristic_extra_features_label_full)
                map_features.update(heuristic_extra_features_label_partial)

            if compute_all:
                if generate_candidate_centerlines > 0:
                    test_extra_features_partial = [additional_centerline_features(test_candidate_centerline_normalized, rotated_and_translated_partial_trajectory, save_str="_PARTIAL") for test_candidate_centerline_normalized in map_features['TEST_CANDIDATE_CENTERLINE_NORMALIZED_PARTIAL']] # NOQA
                    map_features['TEST_CANDIDATE_CENTERLINE_ADDITIONAL_PARTIAL'] = test_extra_features_partial # NOQA
                    if labels is not None:
                        test_extra_features_partial = [additional_centerline_features(test_candidate_centerline_normalized, rotated_and_translated_partial_trajectory, save_str="_FULL") for test_candidate_centerline_normalized in map_features['TEST_CANDIDATE_CENTERLINE_NORMALIZED_FULL']] # NOQA
                        map_features['TEST_CANDIDATE_CENTERLINE_ADDITIONAL_FULL'] = test_extra_features_partial # NOQA
                        extra_features_label_full = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_FULL"], rotated_and_translated_label, save_str="_LABEL_FULL") # NOQA
                        extra_features_label_partial = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"], rotated_and_translated_label, save_str="_LABEL_PARTIAL") # NOQA
                        map_features.update(extra_features_label_full)
                        map_features.update(extra_features_label_partial)
        return map_features

    def compute_social_features(social_agents, map_json, timesteps=20, avm=None, mfu=None, rotation=None,
                                translation=None, generate_candidate_centerlines=0):
        social_features_all = []
        for track_id, social_agent in social_agents.items():
            social_features = OrderedDict([])
            xy_locations = np.array([state.position for state in agent_track.object_states])
            tstamps = np.array([state.timestep for state in agent_track.object_states])
           
            # Remove actors that appear after first 2 seconds
            if tstamps[0] < timesteps:
                # Remove trajectories that are too small
                tsteps = np.sum(tstamps < timesteps)
                if tsteps > 3:
                    labels = xy_locations[tsteps:, :]
                    if len(labels) == 0:
                        labels = None
                    social_features['TSTAMPS'] = tstamps
                    social_features['LABELS_UNNORMALIZED'] = labels

                    # Compute XY Features
                    if xy_features_flag:
                        xy_features = xy_locations[:tsteps, :]
                        if xy_features_normalize_flag:
                            xy_features, _ = normalize_xy(xy_locations=xy_features,
                                                          translation=translation,
                                                          rotation=rotation)
                            if labels is not None:
                                if len(labels) > 1:
                                    social_features["LABELS"] = normalize_xy(xy_locations=labels,
                                                                             translation=translation, # NOQA
                                                                             rotation=rotation)[0]
                                else:
                                    social_features["LABELS"] = normalize_xy(xy_locations=labels,
                                                                             translation=translation, # NOQA
                                                                             rotation=None,
                                                                             to_rotate=False)[0]
                            else:
                                social_features["LABELS"] = np.array([])
                        social_features['XY_FEATURES'] = xy_features

                    # Compute Map Features
                    if map_features_flag:
                        if xy_features_normalize_flag:
                            map_features = compute_map_features(xy_locations=xy_locations,
                                                                map_json=map_json, timesteps=tsteps,
                                                                avm=avm, mfu=mfu, rotation=rotation,
                                                                translation=translation,
                                                                labels=labels,
                                                                generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                                compute_all=False)
                        else:
                            map_features = compute_map_features(xy_locations=xy_locations,
                                                                map_json=map_json, timesteps=tsteps,
                                                                avm=avm, mfu=mfu, labels=labels,
                                                                generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                                compute_all=False)
                        social_features.update(map_features)
                    social_features_all.append(social_features)
        return social_features_all

    '''
    Compute features for current data sequence
    '''
    data = load_argoverse_scenario_hdf5(path)
    map_json = ScenarioMap(path.replace(".h5", "_map.json"))
    final_features = OrderedDict([])
    seq_id = path.split('/')[-1].split('.')[0]
    final_features['PATH'] = os.path.abspath(path)
    final_features['MAP_PATH'] = os.path.abspath(path.replace(".h5", "_map.json"))
    final_features['SEQ_ID'] = data.scenario_id

    all_tracks = {x.track_id: x for x in data.tracks}
    
    # Get focal agent track
    agent_track = all_tracks[data.focal_track_id]
    xy_locations = np.array([state.position for state in agent_track.object_states])
    agent_features = {}

    # Get labels
    labels = None
    if return_labels:
        if label_path == "":
            labels = xy_locations[timesteps:, :]
        else:
            label_data = pd.read_csv(label_path, dtype={"TIMESTAMP": str})
            label_agent_track = label_data[label_data["OBJECT_TYPE"] == "AGENT"]
            labels = label_agent_track[['X', 'Y']].values
            final_features["LABELS_PATH"] = os.path.abspath(label_path)
        agent_features['LABELS_UNNORMALIZED'] = labels
    
    # Get XY input features
    if xy_features_flag:
        xy_features, xy_feature_helpers = compute_xy_features(xy_locations=xy_locations[:timesteps, :], # NOQA
                                                              normalize=xy_features_normalize_flag,
                                                              timesteps=timesteps)
        agent_features['XY_FEATURES'] = xy_features
        if xy_feature_helpers is not None:
            final_features.update(xy_feature_helpers)

    # Compute map features
    if map_features_flag:
        if xy_features_normalize_flag:
            map_features = compute_map_features(xy_locations=xy_locations,
                                                map_json=map_json,
                                                timesteps=timesteps, avm=avm, mfu=mfu,
                                                rotation=xy_feature_helpers['ROTATION'],
                                                translation=xy_feature_helpers['TRANSLATION'],
                                                labels=labels,
                                                generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                compute_all=compute_all)
        else:
            map_features = compute_map_features(xy_locations=xy_locations,
                                                map_json=map_json,
                                                timesteps=timesteps, avm=avm, mfu=mfu,
                                                labels=labels,
                                                generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                compute_all=compute_all)
        agent_features.update(map_features)

    # Compute social features
    if social_features_flag:
        if xy_features_normalize_flag:
            social_features = compute_social_features(social_agents={x:y for x,y in all_tracks.items() if x!=data.focal_track_id}, map_json=map_json,
                                                      timesteps=timesteps, avm=avm, mfu=mfu,
                                                      rotation=xy_feature_helpers['ROTATION'],
                                                      translation=xy_feature_helpers['TRANSLATION'],
                                                      generate_candidate_centerlines=generate_candidate_centerlines) # NOQA
        else:
            social_features = compute_social_features(social_agents={x:y for x,y in all_tracks.items() if x!=data.focal_track_id}, map_json=map_json,
                                                      timesteps=timesteps, avm=avm, mfu=mfu,
                                                      generate_candidate_centerlines=generate_candidate_centerlines) # NOQA
        final_features['SOCIAL'] = social_features

    # Compute Labels
    if return_labels:
        final_features["LABELS"] = normalize_xy(xy_locations=labels,
                                                translation=final_features['TRANSLATION'],
                                                rotation=final_features['ROTATION'])[0]
        agent_features['LABELS'] = final_features['LABELS']

    if bool(agent_features):
        final_features['AGENT'] = agent_features
    return final_features


def cast_dict_f32(old_dict, new_dict):
    """
    Returns a copy of old_dict, with all np arrays cast to float32 in order to save disk space
    """
    for key, item in old_dict.items():
        if '_FULL' in key:
            continue
        if isinstance(item, dict):
            new_dict[key] = cast_dict_f32(item, {})
        elif isinstance(item, list) and isinstance(item[0], dict):
            new_dict[key] = [cast_dict_f32(x, {}) for x in item]
        else:
            if isinstance(item, (np.ndarray, np.int64, np.float64)):
                if item.dtype == np.int64:
                    new_dict[key] = item.astype(np.int32)
                elif item.dtype == np.float64:
                    new_dict[key] = item.astype(np.float32)
            elif isinstance(item, list) and isinstance(item[0], (np.ndarray, np.int64, np.float64)):
                new_dict[key] = [x.astype(np.float32) for x in item]
            else:
                new_dict[key] = item
    return new_dict


def compute_features_wrapper(kwargs):
    """
    Parallelization-friendly wrapper for sequence feature computation
    """
    # Compute features specified by args
    features = compute_features(**kwargs)

    # Cast np arrays in feature dict to f32 to save disk space
    features = cast_dict_f32(features, {})

    # Dump computed features to disk
    with open("{}/{}/{}.pkl".format(args.save_dir, args.mode,
              kwargs['path'].split('/')[-1].split('.')[0]), 'wb') as outFile:
        pkl.dump(features, outFile, -1)

    # Guard against memory leaks by explicitly deleting features
    del features


def compute_features_iterator(path, save_dir, xy_features_flag=True,
                              xy_features_normalize_flag=True, map_features_flag=True,
                              social_features_flag=True, timesteps=20, return_labels=True,
                              generate_candidate_centerlines=0, compute_all=False):
    """
    Compute features for all the trajectories in a directory
        Args:
            path (str): Location of the .csv sequence data files
            xy_features_flag (bool): Use xy coordinate features
            xy_features_normalize_flag (bool): Normalize xy features to constraint start of
                                               sequence to be (0,0) and end of sequence to be on
                                               the positive x axis
            map_features_flag (bool): Compute map features if true
            social_features_flag (bool): Compute social features if true
            timesteps (int): Timesteps for which feature computation needs to be done
                (10 timesteps = 1 second)
            avm (ArgoverseMap object): default None. Pass an object if calling this function in a
                                       loop to avoid redundant computation
            return_labels (bool): Compute the labels for the given sequence
            test_labels_path (str): In case you have access to the test labels
        Returns:
            None: Saves dataframe to file
    """
    # Read data sequence indexes from source directory
    sequences = sorted(os.listdir(path))
    if args.start_file != 0 or args.chunk_length != -1:
        sequences = sequences[args.start_file:(args.start_file + args.chunk_length)]
        print("Start Index:", args.start_file, "End Index:", (args.start_file + args.chunk_length),
              "Num Files:", len(sequences))

    # Create save directory for processed data
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + '/' + args.mode, exist_ok=True)
    avm = ArgoverseMap()
    mfu = MapFeaturesUtils()

    if args.mode == 'test':
        if args.test_labels_path == "":
            return_labels = False

    # Iterate over all files
    input_params = []
    for i, file in enumerate(tqdm(sequences)):
        file_path = os.path.join(path, file)
        label_path = ""
        if args.mode == 'test' and args.test_labels_path != "":
            label_path = os.path.join(args.test_labels_path, file)
        if '.h5' in file_path:
            input_param = {'path': file_path, 'xy_features_flag': xy_features_flag,
                        'xy_features_normalize_flag': xy_features_normalize_flag,
                        'map_features_flag': map_features_flag,
                        'social_features_flag': social_features_flag, 'avm': avm, 'mfu': mfu,
                        'timesteps': timesteps, 'return_labels': return_labels,
                        'label_path': label_path,
                        'generate_candidate_centerlines': generate_candidate_centerlines,
                        'compute_all': compute_all}
        input_params.append(input_param)

    _ = parmap.map(compute_features_wrapper, input_params, pm_pbar=True,
                   pm_processes=args.num_cpus)


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    compute_features_iterator("{}/{}".format(args.dataroot, args.mode), save_dir=args.save_dir,
                              social_features_flag=args.social_features,
                              map_features_flag=args.map_features,
                              xy_features_flag=args.xy_features,
                              xy_features_normalize_flag=args.normalize, timesteps=args.timesteps,
                              generate_candidate_centerlines=args.generate_candidate_centerlines,
                              compute_all=args.compute_all)