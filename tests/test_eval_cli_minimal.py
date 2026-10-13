"""Without the eval extra, openvlharness-eval explains how to install it instead of failing on an import."""
import subprocess
import sys


def test_eval_cli_without_eval_extra_explains_how_to_install():
    script = "import sys; sys.modules['pandas'] = None; from evaluation_kit import cli"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1, proc.stderr
    assert "pip install -e '.[eval]'" in proc.stderr
