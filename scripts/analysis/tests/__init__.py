"""Tests of the OE1 stage-1 tools.

Run them from scripts/analysis without leaving __pycache__ folders behind:

    python -B -m unittest discover -s tests -t .        (or PYTHONDONTWRITEBYTECODE=1)

Importing this package also switches the bytecode cache off for everything imported after it (the test
modules, ``fixtures`` and the tools under test); ``-B`` / the environment variable additionally cover this
file itself.
"""
import sys

sys.dont_write_bytecode = True
