import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent  # -> repo root
sys.path.insert(0, str(ROOT))

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("DISABLE_HEAVY_INIT", "1")

@pytest.fixture(autouse=True)
def _chdir_root(monkeypatch):
    monkeypatch.chdir(ROOT)

