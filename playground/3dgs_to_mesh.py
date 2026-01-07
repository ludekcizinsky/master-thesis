from __future__ import annotations

import os
import sys

import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.helpers.gs_to_mesh import Args, main


if __name__ == "__main__":
    main(tyro.cli(Args))
