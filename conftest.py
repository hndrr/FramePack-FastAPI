import sys
import os

# Add the project root directory to the Python path
# This allows pytest to find modules like 'api'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# You can add fixtures or other pytest configurations here later if needed