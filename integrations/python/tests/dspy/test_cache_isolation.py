"""Run the documented offline program without loading DiskCache."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("dspy")


def test_offline_example_never_loads_diskcache(server, tmp_path):
    example = Path(__file__).resolve().parents[2] / "examples" / "dspy_rag.py"
    code = """
import importlib.abc
import os
from pathlib import Path
import runpy
import sys

attempts = []
class RejectDiskCache(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'diskcache' or fullname.startswith('diskcache.'):
            attempts.append(fullname)
            raise ImportError('DiskCache is forbidden in this test')

sys.meta_path.insert(0, RejectDiskCache())
runpy.run_path(sys.argv[1], run_name='__main__')
import dspy
request = {'prompt': 'local cache round trip'}
assert dspy.cache.get(request) is None
dspy.cache.put(request, {'answer': 'memory only'})
assert dspy.cache.get(request) == {'answer': 'memory only'}
assert attempts == [], attempts
assert not Path(os.environ['DSPY_CACHEDIR']).exists()
"""
    env = {**os.environ, "WARM_URL": server.url, "WARM_TENANT": server.tenant(),
           "WARM_INPUT_DIM": str(server.dim), "DSPY_CACHEDIR": str(tmp_path / "cache"),
           "LITELLM_LOCAL_MODEL_COST_MAP": "True", "WARM_API_KEY": ""}
    result = subprocess.run([sys.executable, "-c", code, str(example)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "on your machine" in result.stdout
