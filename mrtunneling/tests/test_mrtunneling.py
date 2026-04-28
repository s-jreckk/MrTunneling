"""
Unit and regression test for the mrtunneling package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mrtunneling


def test_mrtunneling_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mrtunneling" in sys.modules
