## Features

- Configurable distance thresholds for proximity analysis
- Support for additional location points beyond project data
- Performance optimization options including spatial indexing
- Case-insensitive column name handling with support for various naming formats

## Setup Instructions

### Configuration

This tool uses a `config.yaml` file to specify settings. The first time you run the script, it will create a default configuration file. You can edit this file to customize:

- File paths for input and output
- Distance thresholds for analysis (multiple values supported)
- Additional location points to check
- Performance optimization settings

Example configuration:

```yaml
# Sea Turtle Stranding Analysis Configuration

# Input/Output file paths
files:
  projects: "data/project_summary_export.xlsx"
  turtles: "data/STSSN_report.xlsx"
  output: "output/updated_STSSN_report.xlsx"

# Analysis parameters
analysis:
  # Multiple distance thresholds in kilometers
  distance_thresholds: [5, 10, 15]
  
  # Optional additional locations to check
  additional_locations:
    enabled: false  # Set to true to enable additional location checking
    locations:
      - latitude: 28.3852
        longitude: -80.6052
```

### Installation

#### Option 1: Using pip directly
```bash
# Install required dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

#### Option 2: Using venv (recommended)
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

## Requirements

- Python 3.7+
- pandas
- openpyxl
- pyyaml

## Input Data Format

### Project Data File
Required columns (case-insensitive, supports various formats):
- `dqm_start_date`: Project start date (also supports variations like 'start_date', 'startDate', etc.)
- `dqm_end_date`: Project end date (also supports variations like 'end_date', 'endDate', etc.)
- `dredging_lat`: Latitude of project location (also supports variations like 'lat', 'projectLat', etc.)
- `dredging_lng`: Longitude of project location (also supports variations like 'lng', 'long', etc.)

### Turtle Data File
Required columns (case-insensitive, supports various formats):
- `stssnID`: Unique ID for the turtle (also supports variations like 'stssn_id', 'turtle_id', etc.)
- `reportDate`: Date the turtle was reported (also supports variations like 'report_date', 'date', etc.)
- `latitude`: Latitude where turtle was found (also supports variations like 'lat')
- `longitude`: Longitude where turtle was found (also supports variations like 'lng', 'long')

## Output

The script produces an Excel file with the original turtle data plus additional columns indicating whether each turtle was found within the specified distances of any project active at the time of the stranding.