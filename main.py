import os.path
import json
import sys
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def main():
    # Load configuration
    config = load_config()
    projects_file = config['projects_file']
    turtles_file = config['turtles_file']
    updated_turtles_file = config['output_file']

    # Ensure output directory exists
    os.makedirs(os.path.dirname(updated_turtles_file), exist_ok=True)

    # Check if input files exist
    if not os.path.exists(projects_file):
        print(f"Error: Projects file not found: {projects_file}")
        sys.exit(1)
    if not os.path.exists(turtles_file):
        print(f"Error: Turtles file not found: {turtles_file}")
        sys.exit(1)

    print(f"Using files:")
    print(f"  Projects: {projects_file}")
    print(f"  Turtles: {turtles_file}")
    print(f"  Output: {updated_turtles_file}")

    # Load and validate data
    projects_df, turtles_df_original = load_data(projects_file, turtles_file)

    # Create a working copy and filter out invalid coordinates
    turtles_df = turtles_df_original.copy()
    valid_turtles_df = turtles_df.dropna(subset=['latitude', 'longitude'])

    # Run analysis and get near_project results for both distance thresholds
    near_project_dict_5km, near_project_dict_15km = check_turtle_projects(
        projects_df, valid_turtles_df
    )

    # Update the original dataframe with near_project information for both thresholds
    turtles_df_original['near_project_5km'] = 'No'  # Default value
    turtles_df_original['near_project_15km'] = 'No'  # Default value

    for stssnID, near_status in near_project_dict_5km.items():
        turtles_df_original.loc[turtles_df_original['stssnID'] == stssnID, 'near_project_5km'] = near_status

    for stssnID, near_status in near_project_dict_15km.items():
        turtles_df_original.loc[turtles_df_original['stssnID'] == stssnID, 'near_project_15km'] = near_status

    # Save updated turtle data
    turtles_df_original.to_excel(updated_turtles_file, index=False)
    print(f"Analysis complete. Updated turtle data saved to {updated_turtles_file}")


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


def check_turtle_projects(projects_df, turtles_df, radius_5km=5, radius_15km=15):
    """
    Check if turtles are present near project locations using two distance thresholds.

    Returns:
        - Dictionary mapping turtle IDs to near_project_5km status ('Yes'/'No')
        - Dictionary mapping turtle IDs to near_project_15km status ('Yes'/'No')
    """
    # Dictionaries to track which turtles are near projects at different distances
    near_project_dict_5km = {}
    near_project_dict_15km = {}

    # Initialize all turtles as 'No'
    for _, turtle in turtles_df.iterrows():
        near_project_dict_5km[turtle['stssnID']] = 'No'
        near_project_dict_15km[turtle['stssnID']] = 'No'

    # Filter out projects with invalid coordinates
    valid_projects_df = projects_df.dropna(subset=['dredging_lat', 'dredging_lng'])

    for _, project in valid_projects_df.iterrows():
        # Skip projects with invalid dates
        if pd.isna(project['dqm_start_date']) or pd.isna(project['dqm_end_date']):
            continue

        # Temporal filter
        mask = (
                (turtles_df['reportDate'] >= project['dqm_start_date']) &
                (turtles_df['reportDate'] <= project['dqm_end_date'])
        )
        temp_turtles = turtles_df[mask]

        # Skip if no turtles in time range
        if temp_turtles.empty:
            continue

        # Spatial check for both thresholds
        for _, turtle in temp_turtles.iterrows():
            distance = haversine(
                project['dredging_lat'], project['dredging_lng'],
                turtle['latitude'], turtle['longitude']
            )

            if distance <= radius_5km:
                near_project_dict_5km[turtle['stssnID']] = 'Yes'

            if distance <= radius_15km:
                near_project_dict_15km[turtle['stssnID']] = 'Yes'

    return near_project_dict_5km, near_project_dict_15km


def load_config():
    """Load configuration from config.json or create with defaults if not present."""
    config_path = 'config.json'
    default_config = {
        "projects_file": os.path.join("data", "project_report_summary.xlsx"),
        "turtles_file": os.path.join("data", "STSSN_File.xlsx"),
        "output_file": os.path.join("output", "updated_turtles.xlsx")
    }

    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default config file: {config_path}")

    # Load config
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
    # Earth radius in km
    R = 6371.0

    # Convert to radians
    lat1, lon1 = radians(float(lat1)), radians(float(lon1))
    lat2, lon2 = radians(float(lat2)), radians(float(lon2))

    # Differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


if __name__ == "__main__":
    main()