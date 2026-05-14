"""Logo / favicon assets prepared once at import time.

Streamlit's ``st.set_page_config`` accepts a path string for ``page_icon``,
but the in-page brand image is rendered inside HTML via ``st.markdown`` and
needs a ``data:`` URI. We base64-encode the logo bytes once here so every
view can reuse the result without re-reading the file.

If the logo files are missing (someone cloned without LFS, for example),
``LOGO_URI`` falls back to an empty string and ``FAVICON`` to ``None`` —
all consumers degrade gracefully.
"""

import base64
from pathlib import Path

from config import FAVICON_PATH, LOGO_PATH


def _b64(path: Path) -> str:
    if not path.exists():
        return ''
    try:
        return base64.b64encode(path.read_bytes()).decode('ascii')
    except Exception:
        return ''


_LOGO_B64 = _b64(LOGO_PATH)
LOGO_URI = f'data:image/png;base64,{_LOGO_B64}' if _LOGO_B64 else ''

# Favicon needs a square-ish source so the tab icon is recognizable.
FAVICON = str(FAVICON_PATH) if FAVICON_PATH.exists() else (
    str(LOGO_PATH) if LOGO_PATH.exists() else None
)
