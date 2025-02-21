import os.path
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def main():
    # File paths - modify these to match your actual files
    projects_file = os.path.join("data", "v_project_summary_export_1_3_2025.xlsx")
    turtles_file = os.path.join("data", "20250204_AsbornoUSACE_STSSN.xlsx")
    output_file = "project_presence_results.xlsx"
    updated_turtles_file = "updated_turtles.xlsx"

    # Load and validate data
    projects_df, turtles_df_original = load_data(projects_file, turtles_file)

    # Create a working copy and filter out invalid coordinates
    turtles_df = turtles_df_original.copy()
    valid_turtles_df = turtles_df.dropna(subset=['latitude', 'longitude'])

    # Run analysis and get near_project results
    project_results, near_project_dict = check_turtle_projects(projects_df, valid_turtles_df)

    # Update the original dataframe with near_project information
    turtles_df_original['near_project'] = 'No'  # Default value
    for stssnID, near_status in near_project_dict.items():
        turtles_df_original.loc[turtles_df_original['stssnID'] == stssnID, 'near_project'] = near_status

    # Save results
    project_results.to_excel(output_file, index=False)
    turtles_df_original.to_excel(updated_turtles_file, index=False)
    print(f"Analysis complete. Project results saved to {output_file}")
    print(f"Updated turtle data saved to {updated_turtles_file}")


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


def check_turtle_projects(projects_df, turtles_df, radius_km=5):
    """
    Check if turtles are present near project locations.

    Returns:
        - DataFrame with project results
        - Dictionary mapping turtle IDs to near_project status ('Yes'/'No')
    """
    project_results = []

    # Dictionary to track which turtles are near projects
    near_project_dict = {}

    # Initialize all turtles as 'No'
    for _, turtle in turtles_df.iterrows():
        near_project_dict[turtle['stssnID']] = 'No'

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
            project_results.append({
                'odess_project_id': project['odess_project_id'],
                'project_name': project['project_name'],
                'turtle_present': 'No'
            })
            continue

        # Spatial check
        project_presence = False
        for _, turtle in temp_turtles.iterrows():
            distance = haversine(
                project['dredging_lat'], project['dredging_lng'],
                turtle['latitude'], turtle['longitude']
            )

            if distance <= radius_km:
                project_presence = True
                # Mark this turtle as near a project
                near_project_dict[turtle['stssnID']] = 'Yes'

        project_results.append({
            'odess_project_id': project['odess_project_id'],
            'project_name': project['project_name'],
            'turtle_present': 'Yes' if project_presence else 'No'
        })

    return pd.DataFrame(project_results), near_project_dict


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