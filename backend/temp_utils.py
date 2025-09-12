import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a temporary directory if it doesn't exist
try:
    TEMP_DIR = os.path.join(tempfile.gettempdir(), "hazard_spotter_temp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"Temporary directory created at: {TEMP_DIR}")
except Exception as e:
    logger.error(f"Error creating temporary directory: {e}")
    # Fallback to current directory if temp creation fails
    TEMP_DIR = os.path.join(os.getcwd(), "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.warning(f"Using fallback temporary directory: {TEMP_DIR}")

def get_temp_path(filename):
    """Get a path for a temporary file that works in both development and production"""
    try:
        # Make sure the temp directory exists (might have been deleted)
        os.makedirs(TEMP_DIR, exist_ok=True)
        return os.path.join(TEMP_DIR, filename)
    except Exception as e:
        logger.error(f"Error in get_temp_path: {e}")
        # Last resort fallback - use current directory
        return os.path.join(os.getcwd(), filename)
