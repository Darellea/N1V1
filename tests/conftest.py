# Ensure project root is on sys.path so pytest can import local packages when run
# from arbitrary working directories or when path resolution behaves unexpectedly.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
