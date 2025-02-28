#!/usr/bin/env python3
"""
Sea Turtle Stranding and Salvage Network (STSSN) Analysis Tool

This script analyzes sea turtle stranding data in relation to project locations,
identifying turtles found within configurable distances of active projects.
"""

import os
import sys
import time
from typing import Dict

import pandas as pd
import yaml
from math import radians, sin, cos, sqrt, atan2


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
            return yaml.safe_load(f)
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

        # Load data
        self._load_data()

        # Process data
        self._process_data()

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

    def _load_data(self) -> None:
        """Load and validate data from input files."""
        projects_path = self.config['files']['projects']
        turtles_path = self.config['files']['turtles']

        # Load data
        projects_df = pd.read_excel(projects_path)
        turtles_df = pd.read_excel(turtles_path)

        # Standardize column names to handle variations
        self.projects_df = _standardize_column_names(projects_df)
        self.turtles_df = _standardize_column_names(turtles_df)

        # Validate required columns
        required_project_cols = {
            'dqm_start_date', 'dqm_end_date', 'dredging_lat', 'dredging_lng'
        }
        required_turtle_cols = {
            'stssnid', 'reportdate', 'latitude', 'longitude'
        }

        missing_project_cols = required_project_cols - set(self.projects_df.columns)
        if missing_project_cols:
            raise ValueError(f"Projects spreadsheet missing required columns: {missing_project_cols}")

        missing_turtle_cols = required_turtle_cols - set(self.turtles_df.columns)
        if missing_turtle_cols:
            raise ValueError(f"Turtles spreadsheet missing required columns: {missing_turtle_cols}")

        # Convert dates
        self.projects_df['dqm_start_date'] = pd.to_datetime(self.projects_df['dqm_start_date'])
        self.projects_df['dqm_end_date'] = pd.to_datetime(self.projects_df['dqm_end_date'])
        self.turtles_df['reportdate'] = pd.to_datetime(self.turtles_df['reportdate'])

        # Ensure coordinate columns are numeric
        for col in ['dredging_lat', 'dredging_lng']:
            self.projects_df[col] = pd.to_numeric(self.projects_df[col], errors='coerce')
        for col in ['latitude', 'longitude']:
            self.turtles_df[col] = pd.to_numeric(self.turtles_df[col], errors='coerce')

    def _process_data(self) -> None:
        """Process data and save results."""
        # Get valid data (drop rows with missing coordinates)
        valid_turtles_df = self.turtles_df.dropna(subset=['latitude', 'longitude'])
        valid_projects_df = self.projects_df.dropna(
            subset=['dredging_lat', 'dredging_lng', 'dqm_start_date', 'dqm_end_date'])

        # Get distance thresholds from config
        distance_thresholds = self.config['analysis'].get('distance_thresholds', [5, 10, 15])

        # Initialize results
        threshold_results = {}
        for threshold in distance_thresholds:
            threshold_results[threshold] = self._check_distance_threshold(valid_projects_df, valid_turtles_df,
                                                                          threshold)

        # Add optional additional locations if enabled
        if self.config.get('analysis', {}).get('additional_locations', {}).get('enabled', False):
            self._check_additional_locations(valid_turtles_df, threshold_results)

        # Preserve all original columns from the turtle file
        result_df = self.turtles_df.copy()

        # Add result columns
        for threshold in distance_thresholds:
            column_name = f'near_project_{threshold}km'
            result_df[column_name] = 'No'

            # Update values based on results
            for stssnID, status in threshold_results[threshold].items():
                result_df.loc[result_df['stssnid'] == stssnID, column_name] = status

        # Save results, preserving all original columns
        result_df.to_excel(self.config['files']['output'], index=False)

    def _check_distance_threshold(self, projects_df: pd.DataFrame, turtles_df: pd.DataFrame, threshold: float) -> Dict[
        str, str]:
        """Check if turtles are within the specified distance of projects."""
        print(f"Analyzing turtle proximity within {threshold}km of projects...")

        # Initialize all turtles as 'No'
        results = {turtle_id: 'No' for turtle_id in turtles_df['stssnid']}

        # Use spatial indexing if enabled
        use_spatial_index = self.config.get('performance', {}).get('use_spatial_index', False)
        if use_spatial_index:
            return self._check_with_spatial_index(projects_df, turtles_df, threshold, results)

        # Process each project
        for _, project in projects_df.iterrows():
            # Temporal filter: find turtles reported during project timeframe
            time_mask = (
                    (turtles_df['reportdate'] >= project['dqm_start_date']) &
                    (turtles_df['reportdate'] <= project['dqm_end_date'])
            )
            time_filtered_turtles = turtles_df[time_mask]

            if time_filtered_turtles.empty:
                continue

            # Check distance for each temporally-filtered turtle
            for _, turtle in time_filtered_turtles.iterrows():
                distance = self._haversine(
                    project['dredging_lat'], project['dredging_lng'],
                    turtle['latitude'], turtle['longitude']
                )

                if distance <= threshold:
                    results[turtle['stssnid']] = 'Yes'

        return results

    def _check_with_spatial_index(self, projects_df: pd.DataFrame, turtles_df: pd.DataFrame,
                                  threshold: float, results: Dict[str, str]) -> Dict[str, str]:
        """Check distances using a spatial indexing approach for better performance."""
        batch_size = self.config.get('performance', {}).get('batch_size', 1000)

        # Process in batches
        for i in range(0, len(turtles_df), batch_size):
            batch = turtles_df.iloc[i:i + batch_size]

            for _, project in projects_df.iterrows():
                # Temporal filter
                time_mask = (
                        (batch['reportdate'] >= project['dqm_start_date']) &
                        (batch['reportdate'] <= project['dqm_end_date'])
                )
                time_filtered_batch = batch[time_mask]

                if time_filtered_batch.empty:
                    continue

                # Vectorized distance calculation
                distances = time_filtered_batch.apply(
                    lambda row: self._haversine(
                        project['dredging_lat'], project['dredging_lng'],
                        row['latitude'], row['longitude']
                    ),
                    axis=1
                )

                # Mark turtles within threshold
                nearby_turtles = time_filtered_batch[distances <= threshold]
                for _, turtle in nearby_turtles.iterrows():
                    results[turtle['stssnid']] = 'Yes'

        return results

    def _check_additional_locations(self, turtles_df: pd.DataFrame,
                                    threshold_results: Dict[int, Dict[str, str]]) -> None:
        """Check distances against additional locations specified in config."""
        if not self.config.get('analysis', {}).get('additional_locations', {}).get('locations'):
            return

        print("Checking additional locations...")
        locations = self.config['analysis']['additional_locations']['locations']

        for i, location in enumerate(locations):
            lat = location.get('latitude')
            lng = location.get('longitude')

            if not (lat and lng):
                print(f"Skipping location #{i + 1} due to missing coordinates")
                continue

            print(f"Checking location #{i + 1}: ({lat}, {lng})")

            for threshold, results in threshold_results.items():
                for _, turtle in turtles_df.iterrows():
                    if results.get(turtle['stssnid']) == 'Yes':
                        continue  # Skip turtles already marked as 'Yes'

                    distance = self._haversine(
                        lat, lng,
                        turtle['latitude'], turtle['longitude']
                    )

                    if distance <= threshold:
                        results[turtle['stssnid']] = 'Yes'

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