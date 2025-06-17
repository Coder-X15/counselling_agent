import sys
import os

# Add the project root directory (where this conftest.py is located)
# to the Python path. This allows pytest to find your 'agent' module.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)