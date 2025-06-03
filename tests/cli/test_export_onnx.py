import logging
import pytest
import subprocess

logger = logging.getLogger(__name__)


def test_export_onnx():
    rs = subprocess.run(["python", "yolox/cli/export_onnx.py", "--name", "yolox_s", "--onnx-name", "yolox_s.onnx", "--onnxsim"])
    if rs.returncode != 0:
        pytest.fail("yolox/cli/export_onnx.py failed. See the log for details!")
    rs = subprocess.run(
        ["python", "yolox/cli/export_onnx.py", "--name", "yolox_s", "--onnx-name", "yolox_s.onnx"])
    if rs.returncode != 0:
        pytest.fail("yolox/cli/export_onnx.py failed. See the log for details!")
    rs = subprocess.run(
        ["python", "yolox/cli/export_onnx.py", "--name", "yolox_custom", "--config", "/path/to/yolox_s_custom.cfg", "--onnx-name", "yolox_sc.onnx"])
    if rs.returncode != 0:
        pytest.fail("yolox/cli/export_onnx.py failed. See the log for details!")
