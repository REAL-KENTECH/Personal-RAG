"""pytest bootstrap.

Adds the repository root to ``sys.path`` so tests can do
``from chunking import ...`` without installing the project as a package.
The package layout is flat (no src/ wrapper, no setup.py); pytest's
rootdir detection puts the conftest's directory on sys.path automatically,
but we add an explicit insert here so running pytest from a subdirectory
still works.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
