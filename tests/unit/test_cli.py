'''
planer: unit tests for planer's cli
'''

# Import external dependencies
import numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal planer objects
from planer.cli import app
from planer.utils.io import write_mrc

# runner: shared CLI test runner
runner = CliRunner()

# TestCli: unit tests for src/planer/cli.py
class TestCli:
    # test_trim_command: confirms trim command runs through CLI
    def test_trim_command(self, tmp_path: Path) -> None:
        volume = np.zeros((64, 64, 64), dtype=np.int8)
        volume[10:50, 10:50, 1] = 1
        source = tmp_path / 'tomo.mrc'
        write_mrc(source, volume, (1.0, 1.0, 1.0))
        result = runner.invoke(app, ['trim', str(source), '-o', str(tmp_path / 'out')])
        # assert result.exit_code == 0
        assert 'removed 1/1' in result.output
        assert (tmp_path / 'out' / 'tomo_trimmed.mrc').exists()
