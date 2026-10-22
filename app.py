"""Streamlit Cloud entry point for Character Replace AI.

The full task implementation lives in tasks/character_replace/app.py.
This root-level file lets Streamlit Community Cloud deploy the repository
using its default app path without duplicating the implementation.
"""
from pathlib import Path
import runpy

TASK_APP = Path(__file__).parent / "tasks" / "character_replace" / "app.py"

if not TASK_APP.is_file():
    raise FileNotFoundError(
        f"Character Replace task app was not found: {TASK_APP}"
    )

runpy.run_path(str(TASK_APP))
