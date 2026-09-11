"""``python3 -m gslot`` -- run the arbiter daemon."""

from __future__ import annotations

import sys

from gslot.daemon import main

if __name__ == "__main__":
    sys.exit(main())
