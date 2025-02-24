import os.path
import json
import sys
import time
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def main():
    start_time = time.time()

    config = load_config()
    projects_file = config['projects_file']
    turtles_file = config['turtles_file']
    updated_turtles_file = config['output_file']

    os.makedirs(os.path.dirname(updated_turtles_file), exist_ok=True)

    if not os.path.exists(projects_file):
        print(f"Error: Projects file not found: {projects_file}")
        sys.exit(1)
    if not os.path.exists(turtles_file):
        print(f"Error: Turtles file not found: {turtles_file}")
        sys.exit(1)

    print(f"Using files:")
    print(f"  Projects: {projects_file}")
    print(f"  Turtles: {turtles_file}")
    print(f"  Output: {updated_turtles_file}\n")

    projects_df, turtles_df_original = load_data(projects_file, turtles_file)

    turtles_df = turtles_df_original.copy()
    valid_turtles_df = turtles_df.dropna(subset=['latitude', 'longitude'])

    near_project_dict_5km, near_project_dict_10km, near_project_dict_15km = check_turtle_projects(
        projects_df, valid_turtles_df
    )

    turtles_df_original['near_project_5km'] = 'No'
    turtles_df_original['near_project_10km'] = 'No'
    turtles_df_original['near_project_15km'] = 'No'

    for stssnID, near_status in near_project_dict_5km.items():
        turtles_df_original.loc[turtles_df_original['stssnID'] == stssnID, 'near_project_5km'] = near_status

    for stssnID, near_status in near_project_dict_15km.items():
        turtles_df_original.loc[turtles_df_original['stssnID'] == stssnID, 'near_project_15km'] = near_status

    turtles_df_original.to_excel(updated_turtles_file, index=False)

    end_time = time.time()
    print(f"Analysis complete. Updated STSSN report data saved to {updated_turtles_file}\n")

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}")


def load_data(projects_path, turtles_path):
    """Load data from spreadsheets with validation"""
    projects = pd.read_excel(projects_path)
    turtles = pd.read_excel(turtles_path)

    # Validate required columns
    required_project_cols = {
        'odess_project_id',
        'project_name',
        'dqm_start_date',
        'dqm_end_date',
        'dredging_lat',
        'dredging_lng'
    }
    required_turtle_cols = {
        'stssnID',
        'reportDate',
        'latitude',
        'longitude'
    }

    if not required_project_cols.issubset(projects.columns):
        missing = required_project_cols - set(projects.columns)
        raise ValueError(f"Projects spreadsheet missing required columns: {missing}")
    if not required_turtle_cols.issubset(turtles.columns):
        missing = required_turtle_cols - set(turtles.columns)
        raise ValueError(f"Turtles spreadsheet missing required columns: {missing}")

    # Convert dates
    projects['dqm_start_date'] = pd.to_datetime(projects['dqm_start_date'])
    projects['dqm_end_date'] = pd.to_datetime(projects['dqm_end_date'])
    turtles['reportDate'] = pd.to_datetime(turtles['reportDate'])

    # Ensure coordinate columns are numeric
    for col in ['dredging_lat', 'dredging_lng']:
        projects[col] = pd.to_numeric(projects[col], errors='coerce')
    for col in ['latitude', 'longitude']:
        turtles[col] = pd.to_numeric(turtles[col], errors='coerce')

    return projects, turtles


def check_turtle_projects(projects_df, turtles_df, radius_5km=5, radius_10km=10, radius_15km=15):
    """
    Check if turtles are present near project locations using multiple distance thresholds.

    Returns:
        - Dictionary mapping turtle IDs to near_project_5km status ('Yes'/'No')
        - Dictionary mapping turtle IDs to near_project_10km status ('Yes'/'No')
        - Dictionary mapping turtle IDs to near_project_15km status ('Yes'/'No')
    """
    print(f"Starting analysis for")
    print(f"  Distance: {radius_5km}km")
    print(f"  Distance: {radius_10km}km")
    print(f"  Distance: {radius_15km}km\n")

    near_project_dict_5km = {}
    near_project_dict_10km = {}
    near_project_dict_15km = {}

    # Initialize all turtles as 'No'
    for _, turtle in turtles_df.iterrows():
        near_project_dict_5km[turtle['stssnID']] = 'No'
        near_project_dict_10km[turtle['stssnID']] = 'No'
        near_project_dict_15km[turtle['stssnID']] = 'No'

    # Filter out projects with invalid coordinates
    valid_projects_df = projects_df.dropna(subset=['dredging_lat', 'dredging_lng'])

    for _, project in valid_projects_df.iterrows():
        if pd.isna(project['dqm_start_date']) or pd.isna(project['dqm_end_date']):
            continue

        # Temporal filter
        mask = (
                (turtles_df['reportDate'] >= project['dqm_start_date']) &
                (turtles_df['reportDate'] <= project['dqm_end_date'])
        )
        temp_turtles = turtles_df[mask]

        if temp_turtles.empty:
            continue

        # Spatial check
        for _, turtle in temp_turtles.iterrows():
            distance = haversine(
                project['dredging_lat'], project['dredging_lng'],
                turtle['latitude'], turtle['longitude']
            )

            if distance <= radius_5km:
                near_project_dict_5km[turtle['stssnID']] = 'Yes'

            if distance <= radius_10km:
                near_project_dict_10km[turtle['stssnID']] = 'Yes'

            if distance <= radius_15km:
                near_project_dict_15km[turtle['stssnID']] = 'Yes'

    return near_project_dict_5km, near_project_dict_10km, near_project_dict_15km


def load_config():
    """Load configuration from config.json or create with defaults if not present."""
    config_path = 'config.json'
    default_config = {
        "projects_file": os.path.join("data", "project_report_summary.xlsx"),
        "turtles_file": os.path.join("data", "STSSN_File.xlsx"),
        "output_file": os.path.join("output", "updated_turtles.xlsx"),
        "distance_thresholds": [5, 10, 15]
    }

    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default config file: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration instead.")
        return default_config

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in kilometers"""
    R = 6371.0

    lat1, lon1 = radians(float(lat1)), radians(float(lon1))
    lat2, lon2 = radians(float(lat2)), radians(float(lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


if __name__ == "__main__":
    main()