'''
planer: unit tests for utility functions
'''

# Import external dependencies
import numpy as np, pytest
from pathlib import Path

# Import internal planer objects
from planer.utils.io import read_mrc, write_mrc

# TestIo: tests for src/planer/utils/io.py
class TestIo:
    # test_write_read_equivalence: data and voxel size are equivalent after writing/reading
    def test_write_read_equivalence(self, tmp_path: Path) -> None:
        data = np.arange(24, dtype=np.int8).reshape(2, 3, 4)
        write_mrc(tmp_path / 'a.mrc', data, (2.0, 2.0, 2.0))
        loaded, voxel_size = read_mrc(tmp_path / 'a.mrc')
        assert np.array_equal(loaded, data)
        assert voxel_size == pytest.approx((2.0, 2.0, 2.0))

    # test_refuses_overwrite: existing files are not overwritten unless overwrite=True
    def test_refuses_overwrite(self, tmp_path: Path) -> None:
        data = np.zeros((2, 2, 2), dtype=np.int8)
        write_mrc(tmp_path / 'a.mrc', data, (1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            write_mrc(tmp_path / 'a.mrc', data, (1.0, 1.0, 1.0))
        write_mrc(tmp_path / 'a.mrc', data, (1.0, 1.0, 1.0), overwrite=True)
