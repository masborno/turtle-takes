# Sea Turtle Stranded and Salvaged Network Analysis Tool

## Setup Instructions

### Configuration

This tool uses a `config.json` file to specify file paths. The first time you run the script, it will create a default configuration file with these settings:

```json
{
    "projects_file": "data/{your_project_report_summary}.xlsx",
    "turtles_file": "data/{your_STSSN_file}.xlsx",
    "output_file": "output/{name_of_your_output}.xlsx"
}
```

### Option 1: Using pip directly
```bash
# Install required dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

### Option 2: Using venv (recommended)
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

### Optional: Add dependency checking to your script
```python
##You can add this code at the top of your script to check for missing dependencies:
##Don't forget to add import sys at the top of your file if you use this approach.
def check_dependencies():
    """Check if required packages are installed and install them if missing."""
    required_packages = ['pandas', 'openpyxl']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        print("Or: pip install " + " ".join(missing_packages))
        return False
    return True

# Call this at the beginning of your main function
if not check_dependencies():
    sys.exit(1)

```