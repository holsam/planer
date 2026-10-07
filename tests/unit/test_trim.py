'''
planer: unit tests for volume trimming
'''

# Import external dependencies
import numpy as np, pytest
from pathlib import Path

# Import internal planer objects
from planer.trim import trim_volume
from planer.utils.schema import Severity
from planer.utils.io import read_mrc, write_mrc

# _write_sample: a volume containing an edge slab and a central blob
def _write_sample(path: Path) -> None:
    volume = np.zeros((64, 64, 64), dtype=np.int8)
    volume[10:50, 10:50, 1] = 1
    volume[30:38, 30:38, 30:38] = 1
    write_mrc(path, volume, (7.0, 7.0, 7.0))

# TestTrim: unit tests for src/planer/trim.py
class TestTrim:
    # test_trim_removes_slab_only: trim_volume removes the edge slab but not the central blob
    def test_trim_removes_slab_only(self, tmp_path: Path) -> None:
        source = tmp_path / 'tomo.mrc'
        _write_sample(source)
        report = trim_volume(source, tmp_path / 'out', Severity.MEDIUM)
        cleaned, voxel_size = read_mrc(report.output)
        assert (report.total, report.removed) == (2, 1)
        assert cleaned[:, :, 1].sum() == 0
        assert cleaned[30:38, 30:38, 30:38].sum() == 512
        assert voxel_size == pytest.approx((7.0, 7.0, 7.0))
        assert len(report.scores.read_text().splitlines()) == 3

    # test_refuses_overwrite_unless_explicit: existing files are not overwritten unless given explicit overwrite=True
    def test_refuses_overwrite_unless_explicit(self, tmp_path: Path) -> None:
        source = tmp_path / 'tomo.mrc'
        _write_sample(source)
        trim_volume(source, tmp_path / 'out', Severity.MEDIUM)
        with pytest.raises(FileExistsError):
            trim_volume(source, tmp_path / 'out', Severity.MEDIUM)
        trim_volume(source, tmp_path / 'out', Severity.MEDIUM, overwrite=True)
