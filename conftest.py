"""
pytest configuration: makes `src/` importable without an editable install.
"""

import os
import sys

# Add src/ to path so `from velvet.sampling import ...` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
