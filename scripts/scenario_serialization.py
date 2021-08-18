"""Utilities to (de)serialize Argoverse forecasting scenarios."""

import json
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import pandas as pd

import h5py
from data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    ScenarioOutputs,
    StaticMapElements,
    Track,
    TrackCategory,
)


def serialize_scenario_outputs(save_dir: Path, scenario_outputs: ScenarioOutputs,
                               scenario_stem: str) -> Tuple[Path, Path]:
    """Serialize all outputs associated with an Argoverse scenario and save to disk.
    Args:
        save_dir: Directory where serialized scenario data should be saved.
        scenario_outputs: All outputs to serialize for a particular Argoverse scenario.
        scenario_stem: Prefix for the serialized file names.
    Returns:
        scenario_path: Path to serialized Argoverse scenario
        static_map_path: Path to serialized static map for scenario
    """
    scenario_path = Path(save_dir) / f"{scenario_stem}.h5"
    static_map_path = Path(save_dir) / f"{scenario_stem}_map.json"

    serialize_argoverse_scenario_hdf5(scenario_path, scenario_outputs.scenario)
    serialize_static_map_json(static_map_path, scenario_outputs.static_map)

    return (scenario_path, static_map_path)


def serialize_argoverse_scenario_hdf5(save_path: Path, scenario: ArgoverseScenario) -> None:
    """Serialize a single Argoverse scenario in HDF5 format and save to disk.
    Args:
        save_path: Path where the scenario should be saved.
        scenario: Scenario to serialize and save.
    """

    # Convert scenario data to DataFrame to reap advantages of HDF5-serialization optimizations.
    tracks_df = _convert_tracks_to_tabular_format(scenario.tracks)

    # Serialize the scenario dataframe as H5 dataset
    with pd.HDFStore(save_path, mode="w") as scenario_data:
        tracks_df.to_hdf(scenario_data, key="tracks_df")

    # Serialize all scenario-associated metadata as H5 attributes
    with h5py.File(save_path, mode="a") as scenario_data:
        # Save scenario-level metadata to attributes
        scenario_data.attrs["scenario_id"] = scenario.scenario_id
        scenario_data.attrs["timestamps"] = scenario.timestamps
        scenario_data.attrs["focal_track_id"] = scenario.focal_track_id
        scenario_data.attrs["map_id"] = scenario.map_id
        scenario_data.attrs["slice_id"] = scenario.slice_id


def load_argoverse_scenario_hdf5(scenario_path: Path) -> ArgoverseScenario:
    """Loads a serialized Argoverse scenario from disk.
    Args:
        scenario_path: Path to the saved scenario.
    Raises:
        FileNotFoundError: If no file exists at the specified scenario_path.
    Returns:
        scenario: Preprocessed scenario object that was saved at scenario_path.
    """
    if not Path(scenario_path).exists():
        raise FileNotFoundError(f"No scenario exists at location: {scenario_path}.")

    # Load scenario dataframe from H5 datasets
    tracks_df = pd.read_hdf(scenario_path, key="tracks_df", mode="r")

    # Load scenario data from DataFrame to reap advantages of HDF5-serialization optimizations.
    tracks = _load_tracks_from_tabular_format(tracks_df)

    # Load metadata from H5 attributes
    with h5py.File(scenario_path, "r") as scenario_data:

        # Load scenario-level metadata
        scenario_id = scenario_data.attrs["scenario_id"]
        timestamps = np.array(scenario_data.attrs["timestamps"])
        focal_track_id = scenario_data.attrs["focal_track_id"]
        map_id = scenario_data.attrs["map_id"]
        slice_id = scenario_data.attrs["slice_id"]

    return ArgoverseScenario(scenario_id=scenario_id,
                             timestamps=timestamps,
                             tracks=tracks,
                             focal_track_id=focal_track_id,
                             map_id=map_id,
                             slice_id=slice_id)


def serialize_static_map_json(save_path: Path, static_map: StaticMapElements) -> None:
    """Serialize a single static map associated with an Argoverse scenario in JSON format and save to disk.
    Args:
        save_path: Path where the static map should be saved.
        static_map: Static map to serialize and save.
    """
    with open(save_path, "w") as f:
        json.dump(static_map, f)


def load_static_map_json(static_map_path: Path) -> StaticMapElements:
    """Load a saved static map from disk.
    Args:
        static_map_path: Path to the saved static map.
    Returns:
        Object representation for the static map elements saved at `static_map_path`.
    """
    with open(static_map_path, "rb") as f:
        static_map_elements = json.load(f)

    return cast(StaticMapElements, static_map_elements)


def _convert_tracks_to_tabular_format(tracks: List[Track]) -> pd.DataFrame:
    """Convert tracks to tabular data format.
    
    Args:
        tracks: All tracks associated with the scenario.
    
    Returns:
        DataFrame containing all track data in a tabular format.
    """

    track_dfs: List[pd.DataFrame] = []

    for track in tracks:

        track_df = pd.DataFrame()

        observed_states: List[bool] = []
        timesteps: List[int] = []
        positions_x: List[float] = []
        positions_y: List[float] = []
        headings: List[float] = []
        velocities_x: List[float] = []
        velocities_y: List[float] = []

        for object_state in track.object_states:
            observed_states.append(object_state.observed)
            timesteps.append(object_state.timestep)
            positions_x.append(object_state.position[0])
            positions_y.append(object_state.position[1])
            headings.append(object_state.heading)
            velocities_x.append(object_state.velocity[0])
            velocities_y.append(object_state.velocity[1])

        track_df["observed"] = observed_states
        track_df["track_id"] = track.track_id
        track_df["object_type"] = track.object_type.value
        track_df["object_category"] = track.category.value
        track_df["timestep"] = timesteps
        track_df["position_x"] = positions_x
        track_df["position_y"] = positions_y
        track_df["heading"] = headings
        track_df["velocity_x"] = velocities_x
        track_df["velocity_y"] = velocities_y

        track_dfs.append(track_df)

    return pd.concat(track_dfs, ignore_index=True)


def _load_tracks_from_tabular_format(tracks_df: pd.DataFrame) -> List[Track]:
    """Load tracks from tabular data format.
    
    Args:
        tracks_df: DataFrame containing all track data in a tabular format.
    
    Returns:
        All tracks associated with the scenario.
    """

    tracks: List[Track] = []

    for track_id, track_df in tracks_df.groupby("track_id"):

        observed_states: List[bool] = track_df.loc[:, "observed"].values.tolist()
        object_type: ObjectType = ObjectType(track_df["object_type"].iloc[0])
        object_category: TrackCategory = TrackCategory(track_df["object_category"].iloc[0])
        timesteps: List[int] = track_df.loc[:, "timestep"].values.tolist()
        positions: List[Tuple[float, float]] = list(
            zip(
                track_df.loc[:, "position_x"].values.tolist(),
                track_df.loc[:, "position_y"].values.tolist(),
            ))
        headings: List[float] = track_df.loc[:, "heading"].values.tolist()
        velocities: List[Tuple[float, float]] = list(
            zip(
                track_df.loc[:, "velocity_x"].values.tolist(),
                track_df.loc[:, "velocity_y"].values.tolist(),
            ))

        object_states: List[ObjectState] = []
        for idx in range(len(timesteps)):
            object_states.append(
                ObjectState(observed=observed_states[idx],
                            timestep=timesteps[idx],
                            position=positions[idx],
                            heading=headings[idx],
                            velocity=velocities[idx]))

        tracks.append(
            Track(track_id=track_id, object_states=object_states, object_type=object_type, category=object_category))

    return tracks
