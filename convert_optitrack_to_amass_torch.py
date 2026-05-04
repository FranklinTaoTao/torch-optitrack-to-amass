#!/usr/bin/env python3
"""Thin entrypoint for the Torch OptiTrack-to-AMASS converter.

The implementation lives under ``torch_optitrack_to_amass/src`` so the Stage-I,
Stage-II, IO, pipeline, and CLI pieces can be read independently. This module
keeps the old import path and command path working.
"""

from __future__ import annotations

if __package__:
    from .src.helpers import *  # noqa: F401,F403
    from .src.helpers import (  # noqa: F401
        _ensure_parent,
        _read_json,
        _repo_root,
        _resolve_mocap_unit,
        _sanitize_stem,
        _torch_device,
    )
    from .src.stagei import optimize_stagei  # noqa: F401
    from .src.stageii import optimize_stageii  # noqa: F401
    from .src.io import save_stagei, save_stageii  # noqa: F401
    from .src.pipeline import run_conversion  # noqa: F401
    from .src.cli import build_parser, main  # noqa: F401
else:
    from src.helpers import *  # noqa: F401,F403
    from src.helpers import (  # noqa: F401
        _ensure_parent,
        _read_json,
        _repo_root,
        _resolve_mocap_unit,
        _sanitize_stem,
        _torch_device,
    )
    from src.stagei import optimize_stagei  # noqa: F401
    from src.stageii import optimize_stageii  # noqa: F401
    from src.io import save_stagei, save_stageii  # noqa: F401
    from src.pipeline import run_conversion  # noqa: F401
    from src.cli import build_parser, main  # noqa: F401


if __name__ == "__main__":
    main()
