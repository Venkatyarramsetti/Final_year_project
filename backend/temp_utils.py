import os
import tempfile

# Create a temporary directory if it doesn't exist
TEMP_DIR = os.path.join(tempfile.gettempdir(), "hazard_spotter_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def get_temp_path(filename):
    """Get a path for a temporary file that works in both development and production"""
    return os.path.join(TEMP_DIR, filename)
