#!/usr/bin/env python3
"""
Sea Turtle Stranding and Salvage Network (STSSN) Analysis Tool

This script analyzes sea turtle stranding data in relation to project locations,
identifying turtles found within configurable distances of active projects,
including turtles found up to 10 days after project completion.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Union
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import yaml
from math import radians, sin, cos, sqrt, atan2

# Import optional dependencies
try:
    from sklearn.neighbors import BallTree

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found. Some optimizations will be disabled.")


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to handle variations in capitalization and naming formats.

    This handles common variations like:
    - Different capitalization (Latitude vs latitude)
    - Underscores vs camelCase (report_date vs reportDate)
    - Spaces (Report Date vs ReportDate)
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # First convert all to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Define mapping for common variations (add more as needed)
    column_mappings = {
        # Project columns
        'dqmstartdate': 'dqm_start_date',
        'dqm start date': 'dqm_start_date',
        'startdate': 'dqm_start_date',
        'start_date': 'dqm_start_date',
        'start date': 'dqm_start_date',

        'dqmenddate': 'dqm_end_date',
        'dqm end date': 'dqm_end_date',
        'enddate': 'dqm_end_date',
        'end_date': 'dqm_end_date',
        'end date': 'dqm_end_date',

        'dredginglat': 'dredging_lat',
        'dredging latitude': 'dredging_lat',
        'projectlat': 'dredging_lat',
        'project_lat': 'dredging_lat',
        'project latitude': 'dredging_lat',

        'dredginglng': 'dredging_lng',
        'dredging longitude': 'dredging_lng',
        'projectlng': 'dredging_lng',
        'project_lng': 'dredging_lng',
        'project longitude': 'dredging_lng',

        # Turtle columns
        'stssn_id': 'stssnid',
        'stssn id': 'stssnid',

        'report_date': 'reportdate',
        'report date': 'reportdate',
        'date': 'reportdate',

        'lat': 'latitude',
        'lng': 'longitude',
        'long': 'longitude',
    }

    # Apply mappings
    df.columns = [column_mappings.get(col, col) for col in df.columns]

    return df


def _load_config(config_path: str) -> dict:
    """Load configuration from YAML file or create with defaults if not present."""
    default_config = {
        "files": {
            "projects": "data/project_summary_export.xlsx",
            "turtles": "data/STSSN_report.xlsx",
            "output": "output/updated_STSSN_report.xlsx"
        },
        "analysis": {
            "distance_thresholds": [5, 10, 15]
        },
        "performance": {
            "use_spatial_index": True,
            "use_parallel": True,
            "batch_size": 1000
        }
    }

    if not os.path.exists(config_path):
        # Create default config if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        print(f"Created default config file: {config_path}")
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

            # Ensure performance section exists
            if 'performance' not in config:
                config['performance'] = default_config['performance']

            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration instead.")
        return default_config


class TurtleAnalyzer:
    """Analyzes sea turtle locations in relation to projects."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the analyzer with configuration settings."""
        self.config = _load_config(config_path)
        self.projects_df = None
        self.turtles_df = None

    def run(self) -> None:
        """Run the full analysis pipeline."""
        start_time = time.time()

        # Ensure output directory exists
        output_path = self.config['files']['output']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Validate input files
        self._validate_files()

        # Load data with optimized method
        self._load_data_optimized()

        # Process data with optimized method
        self._process_data_optimized()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Analysis complete. Updated STSSN report saved to {output_path}")
        print(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}")

    def _validate_files(self) -> None:
        """Validate that required input files exist."""
        projects_file = self.config['files']['projects']
        turtles_file = self.config['files']['turtles']

        if not os.path.exists(projects_file):
            print(f"Error: Projects file not found: {projects_file}")
            sys.exit(1)
        if not os.path.exists(turtles_file):
            print(f"Error: Turtles file not found: {turtles_file}")
            sys.exit(1)

        print(f"Using files:")
        print(f"  Projects: {projects_file}")
        print(f"  Turtles: {turtles_file}")
        print(f"  Output: {self.config['files']['output']}\n")

    def _get_column_mapping(self, headers: List[str], required_cols: List[str]) -> Dict[str, str]:
        """Map actual file headers to our standardized column names."""
        header_to_std = {}

        # Convert headers to lowercase for matching
        headers_lower = [h.lower() if h is not None else "" for h in headers]

        # Check for each required column
        for std_col in required_cols:
            found = False

            # Try exact match first
            if std_col in headers_lower:
                idx = headers_lower.index(std_col)
                header_to_std[headers[idx]] = std_col
                found = True
                continue

            # Try common variations based on our standardization logic
            if std_col == 'dqm_start_date':
                variations = ['startdate', 'start_date', 'start date', 'dqmstartdate', 'dqm start date',
                              'beginning date']
            elif std_col == 'dqm_end_date':
                variations = ['enddate', 'end_date', 'end date', 'dqmenddate', 'dqm end date', 'finish date']
            elif std_col == 'dredging_lat':
                variations = ['lat', 'latitude', 'projectlat', 'project_lat', 'project latitude', 'dredginglat']
            elif std_col == 'dredging_lng':
                variations = ['lng', 'long', 'longitude', 'projectlng', 'project_lng', 'project longitude',
                              'dredginglng']
            elif std_col == 'stssnid':
                variations = ['stssn_id', 'stssn id', 'id', 'turtle_id']
            elif std_col == 'reportdate':
                variations = ['report_date', 'report date', 'date', 'observation date']
            elif std_col == 'latitude':
                variations = ['lat', 'turtle_lat', 'turtle latitude']
            elif std_col == 'longitude':
                variations = ['lng', 'long', 'turtle_lng', 'turtle longitude']
            else:
                variations = []

            # Check variations
            for var in variations:
                if var in headers_lower:
                    idx = headers_lower.index(var)
                    header_to_std[headers[idx]] = std_col
                    found = True
                    break

            if not found:
                # Try partial matches
                for i, h in enumerate(headers_lower):
                    if std_col.replace('_', '') in h.replace('_', '').replace(' ', ''):
                        header_to_std[headers[i]] = std_col
                        found = True
                        break

            if not found:
                raise ValueError(f"Could not find column mapping for required column: {std_col}")

        return header_to_std

    def _load_data_optimized(self) -> None:
        """Load and validate data from input files with performance optimizations."""
        projects_path = self.config['files']['projects']
        turtles_path = self.config['files']['turtles']

        # Define required columns
        project_cols = ['dqm_start_date', 'dqm_end_date', 'dredging_lat', 'dredging_lng']
        turtle_cols = ['stssnid', 'reportdate', 'latitude', 'longitude']

        try:
            # Load projects DataFrame
            projects_df = pd.read_excel(projects_path)

            # Standardize column names
            projects_df = _standardize_column_names(projects_df)

            # Validate required columns exist after standardization
            missing_project_cols = set(project_cols) - set(projects_df.columns)
            if missing_project_cols:
                raise ValueError(f"Projects spreadsheet missing required columns: {missing_project_cols}")

            # Convert dates to datetime
            projects_df['dqm_start_date'] = pd.to_datetime(projects_df['dqm_start_date'])
            projects_df['dqm_end_date'] = pd.to_datetime(projects_df['dqm_end_date'])

            # Convert coordinates to numeric
            for col in ['dredging_lat', 'dredging_lng']:
                projects_df[col] = pd.to_numeric(projects_df[col], errors='coerce')

            # Filter out projects with missing required data
            valid_projects = ~projects_df[project_cols].isna().any(axis=1)
            if (~valid_projects).sum() > 0:
                print(f"Warning: Dropping {(~valid_projects).sum()} projects with missing required data")
                projects_df = projects_df[valid_projects].copy()

            # Optimize memory usage
            projects_df = projects_df.reset_index(drop=True)

            # Load turtles DataFrame
            turtles_df = pd.read_excel(turtles_path)

            # Standardize column names
            turtles_df = _standardize_column_names(turtles_df)

            # Validate required columns exist after standardization
            missing_turtle_cols = set(turtle_cols) - set(turtles_df.columns)
            if missing_turtle_cols:
                raise ValueError(f"Turtles spreadsheet missing required columns: {missing_turtle_cols}")

            # Convert dates to datetime
            turtles_df['reportdate'] = pd.to_datetime(turtles_df['reportdate'])

            # Convert coordinates to numeric
            for col in ['latitude', 'longitude']:
                turtles_df[col] = pd.to_numeric(turtles_df[col], errors='coerce')

            # Filter out turtles with missing required data
            valid_turtles = ~turtles_df[turtle_cols].isna().any(axis=1)
            if (~valid_turtles).sum() > 0:
                print(f"Warning: Dropping {(~valid_turtles).sum()} turtles with missing required data")
                turtles_df = turtles_df[valid_turtles].copy()

            # Optimize memory usage
            turtles_df = turtles_df.reset_index(drop=True)

            # Store the DataFrames
            self.projects_df = projects_df
            self.turtles_df = turtles_df

            print(f"Loaded {len(self.projects_df)} projects and {len(self.turtles_df)} turtles")

        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def _vectorized_haversine(self, lat1: float, lon1: float, lat2_array: np.ndarray,
                              lon2_array: np.ndarray) -> np.ndarray:
        """
        Vectorized Haversine formula to calculate distances between one point and an array of points.

        Args:
            lat1, lon1: Coordinates of the single point
            lat2_array, lon2_array: NumPy arrays of coordinates to compare against

        Returns:
            NumPy array of distances in kilometers
        """
        R = 6371.0  # Earth radius in kilometers

        # Convert all coordinates to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2_array)
        lon2_rad = np.radians(lon2_array)

        # Calculate differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Apply Haversine formula
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def _build_spatial_index(self, turtles_df: pd.DataFrame) -> Optional[BallTree]:
        """Build a spatial index for faster proximity queries using BallTree."""
        if not HAS_SKLEARN:
            return None

        # Convert coordinates to radians for the BallTree
        coords = np.radians(turtles_df[['latitude', 'longitude']].values)

        # Create a BallTree spatial index
        return BallTree(coords, metric='haversine')

    def _process_project_chunk(self, projects_chunk: pd.DataFrame, turtles_df: pd.DataFrame,
                               threshold: float) -> Dict[str, str]:
        """Process a chunk of projects against all turtles for parallel processing."""
        # Initialize results dictionary for this chunk
        chunk_results = {}

        # Get turtle data as numpy arrays for performance
        turtle_ids = turtles_df['stssnid'].values
        turtle_lats = turtles_df['latitude'].values
        turtle_lons = turtles_df['longitude'].values
        turtle_dates = np.array([d.timestamp() for d in turtles_df['reportdate']])

        # Process each project in the chunk
        for _, project in projects_chunk.iterrows():
            # Convert project dates to timestamps for faster comparison
            start_ts = project['dqm_start_date'].timestamp()
            end_ts = (project['dqm_end_date'] + pd.Timedelta(days=10)).timestamp()

            # Vectorized temporal filtering
            time_mask = (turtle_dates >= start_ts) & (turtle_dates <= end_ts)

            # Skip if no turtles in this time range
            if not np.any(time_mask):
                continue

            # Filter turtles by time
            time_indices = np.where(time_mask)[0]
            filtered_ids = turtle_ids[time_indices]
            filtered_lats = turtle_lats[time_indices]
            filtered_lons = turtle_lons[time_indices]

            # Vectorized distance calculation
            distances = self._vectorized_haversine(
                project['dredging_lat'], project['dredging_lng'],
                filtered_lats, filtered_lons
            )

            # Find turtles within threshold
            nearby_indices = np.where(distances <= threshold)[0]

            # Add to results
            for i in nearby_indices:
                turtle_id = filtered_ids[i]
                chunk_results[turtle_id] = 'Yes'

        return chunk_results

    def _check_distance_threshold_vectorized(self, projects_df: pd.DataFrame, turtles_df: pd.DataFrame,
                                             threshold: float) -> Dict[str, str]:
        """Check if turtles are within the specified distance of projects using vectorized operations."""
        print(f"Analyzing turtle proximity within {threshold}km of projects using vectorized operations...")

        # Initialize all turtles as 'No'
        results = {turtle_id: 'No' for turtle_id in turtles_df['stssnid']}

        # Create NumPy arrays of turtle coordinates for vectorized calculation
        turtle_lats = turtles_df['latitude'].values
        turtle_lons = turtles_df['longitude'].values
        turtle_ids = turtles_df['stssnid'].values
        turtle_dates = turtles_df['reportdate'].values

        # Process each project - this loop cannot be easily vectorized due to date filtering
        for _, project in projects_df.iterrows():
            # Temporal filter using NumPy operations
            project_start = project['dqm_start_date']
            project_end_plus_10 = project['dqm_end_date'] + pd.Timedelta(days=10)

            # Create a mask for turtles in the date range
            time_mask = (turtle_dates >= project_start) & (turtle_dates <= project_end_plus_10)

            # Skip if no turtles in time range
            if not np.any(time_mask):
                continue

            # Get the indices of turtles that are in the time range
            time_indices = np.where(time_mask)[0]

            # Extract coordinates for turtles in the time range
            filtered_lats = turtle_lats[time_indices]
            filtered_lons = turtle_lons[time_indices]
            filtered_ids = turtle_ids[time_indices]

            # Calculate distances using vectorized Haversine
            distances = self._vectorized_haversine(
                project['dredging_lat'], project['dredging_lng'],
                filtered_lats, filtered_lons
            )

            # Find turtles within threshold
            within_threshold = distances <= threshold

            # Update results for turtles within threshold
            for i in np.where(within_threshold)[0]:
                results[filtered_ids[i]] = 'Yes'

        return results

    def _check_with_spatial_index(self, projects_df: pd.DataFrame, turtles_df: pd.DataFrame,
                                  threshold: float, results: Dict[str, str]) -> Dict[str, str]:
        """Check distances using an efficient BallTree spatial index for optimal performance."""
        if not HAS_SKLEARN:
            print("BallTree not available. Falling back to vectorized calculation.")
            return self._check_distance_threshold_vectorized(projects_df, turtles_df, threshold)

        print(f"Analyzing turtle proximity within {threshold}km of projects using BallTree spatial index...")

        # Earth radius in kilometers
        R = 6371.0

        # Convert threshold to radians
        threshold_rad = threshold / R

        # Create a copy of turtles_df with an index column to track original indices
        indexed_turtles = turtles_df.reset_index().rename(columns={'index': 'original_idx'})

        # Build spatial index
        tree = self._build_spatial_index(indexed_turtles)

        # Process each project
        for _, project in projects_df.iterrows():
            # Temporal filter - with 10-day extension
            time_mask = (
                    (indexed_turtles['reportdate'] >= project['dqm_start_date']) &
                    (indexed_turtles['reportdate'] <= project['dqm_end_date'] + pd.Timedelta(days=10))
            )

            # Skip if no turtles in the time range
            if not time_mask.any():
                continue

            # Get temporally filtered turtles
            time_filtered = indexed_turtles[time_mask]

            # Skip if no turtles after filtering
            if time_filtered.empty:
                continue

            # Query point (the project location)
            query_point = np.radians([[project['dredging_lat'], project['dredging_lng']]])

            # Build spatial index for the filtered turtles
            filtered_tree = self._build_spatial_index(time_filtered)

            # Query the spatial index for turtles within the threshold
            indices = filtered_tree.query_radius(query_point, threshold_rad)[0]

            # Update results
            for idx in indices:
                turtle_id = time_filtered.iloc[idx]['stssnid']
                results[turtle_id] = 'Yes'

        return results

    def _check_distance_threshold_parallel(self, projects_df: pd.DataFrame, turtles_df: pd.DataFrame,
                                           threshold: float) -> Dict[str, str]:
        """Check distances in parallel across multiple processes."""
        print(f"Analyzing turtle proximity within {threshold}km of projects using parallel processing...")

        # Initialize all turtles as 'No'
        results = {turtle_id: 'No' for turtle_id in turtles_df['stssnid']}

        # Determine optimal chunk size based on CPU count
        cpu_count = mp.cpu_count()
        chunk_size = max(1, len(projects_df) // (cpu_count * 2))  # Double the CPU count for smaller chunks

        # Create project chunks
        project_chunks = [projects_df.iloc[i:i + chunk_size] for i in range(0, len(projects_df), chunk_size)]

        print(f"Processing {len(projects_df)} projects in {len(project_chunks)} chunks using {cpu_count} processes")

        # Create a pool of worker processes
        with mp.Pool(processes=cpu_count) as pool:
            # Create a partial function with fixed arguments
            process_func = partial(self._process_project_chunk, turtles_df=turtles_df, threshold=threshold)

            # Process chunks in parallel and collect results
            chunk_results_list = pool.map(process_func, project_chunks)

        # Combine results from all chunks
        for chunk_results in chunk_results_list:
            for turtle_id, status in chunk_results.items():
                if status == 'Yes':
                    results[turtle_id] = 'Yes'

        return results


    def _process_data_optimized(self) -> None:
        """Optimized process data method that uses the most appropriate algorithm based on data size."""
        print("Using optimized processing method...")

        # Get distance thresholds from config
        distance_thresholds = self.config['analysis'].get('distance_thresholds', [5, 10, 15])

        # Determine the best processing method based on data size and configuration
        large_dataset = len(self.turtles_df) > 10000 or len(self.projects_df) > 100

        # Get performance settings from config
        performance_config = self.config.get('performance', {})
        use_parallel = performance_config.get('use_parallel', True)
        use_spatial_index = performance_config.get('use_spatial_index', True)

        # Initialize results
        threshold_results = {}

        # For each threshold, choose the best algorithm
        for threshold in distance_thresholds:
            if large_dataset and use_parallel and mp.cpu_count() > 1:
                print(f"Using parallel processing for {threshold}km threshold")
                threshold_results[threshold] = self._check_distance_threshold_parallel(
                    self.projects_df, self.turtles_df, threshold)
            elif large_dataset and HAS_SKLEARN and use_spatial_index:
                print(f"Using BallTree spatial index for {threshold}km threshold")
                threshold_results[threshold] = self._check_with_spatial_index(
                    self.projects_df, self.turtles_df, threshold,
                    {turtle_id: 'No' for turtle_id in self.turtles_df['stssnid']})
            else:
                print(f"Using vectorized calculation for {threshold}km threshold")
                threshold_results[threshold] = self._check_distance_threshold_vectorized(
                    self.projects_df, self.turtles_df, threshold)

        # Create result dataframe
        result_df = self.turtles_df.copy()

        # Add result columns
        for threshold in distance_thresholds:
            column_name = f'near_project_{threshold}km'
            result_df[column_name] = 'No'

            # Update values based on results
            for stssnID, status in threshold_results[threshold].items():
                result_df.loc[result_df['stssnid'] == stssnID, column_name] = status

        # Save results
        output_path = self.config['files']['output']

        # Determine if we should use a fast writer for large datasets
        try:
            import xlsxwriter
            has_xlsxwriter = True
        except ImportError:
            has_xlsxwriter = False

        if has_xlsxwriter and len(result_df) > 100000:
            print(f"Using optimized Excel writer for large dataset ({len(result_df)} rows)")
            with pd.ExcelWriter(output_path, engine='xlsxwriter', options={'constant_memory': True}) as writer:
                result_df.to_excel(writer, index=False, sheet_name='Results')
        else:
            result_df.to_excel(output_path, index=False)

        print(f"Results saved to {output_path}")

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS points in kilometers."""
        R = 6371.0  # Earth radius in kilometers

        lat1, lon1 = radians(float(lat1)), radians(float(lon1))
        lat2, lon2 = radians(float(lat2)), radians(float(lon2))

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Haversine formula
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c


def main():
    try:
        analyzer = TurtleAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())