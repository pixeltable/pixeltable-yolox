import subprocess
import sys
from pathlib import Path

import pytest

EXPORT_SCRIPT = Path(__file__).parents[2] / 'yolox' / 'cli' / 'export_onnx.py'


@pytest.mark.parametrize('extra_args', [['--onnxsim'], []])
def test_export_onnx(tmp_path: Path, extra_args: list[str]) -> None:
    onnx_path = tmp_path / 'yolox_s.onnx'
    cmd = [sys.executable, str(EXPORT_SCRIPT), '--name', 'yolox_s', '--onnx-name', str(onnx_path), *extra_args]
    rs = subprocess.run(cmd, check=False)
    assert rs.returncode == 0, 'yolox/cli/export_onnx.py failed. See the log for details!'
    assert onnx_path.exists()
