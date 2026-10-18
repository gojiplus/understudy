"""Consumer imports must not depend on test-only packages."""

import subprocess
import sys


def test_public_import_without_pytest():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.modules['pytest'] = None; "
            "from understudy import AssertionHelpers, run_turn",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
