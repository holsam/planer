'''
planer: unit tests for utility functions
'''

# Import external dependencies
import numpy as np, pytest
from pathlib import Path

# Import internal planer objects
from planer.utils.components import extract_components
from planer.utils.io import read_mrc, write_mrc

# TestComponents: tests for src/planer/utils/components.py
class TestComponents:
    # test_separate_blobs: two disjointed regions give two components
    def test_separate_blobs_return_different_components(self) -> None:
        volume = np.zeros((20, 20, 20), dtype=np.int8)
        volume[2:4, 2:4, 2:4] = 1
        volume[10:13, 10:13, 10:13] = 1
        components = extract_components(volume)
        assert [len(c.coords) for c in components] == [8, 27]
        assert components[1].coords.min(axis=0).tolist() == [10, 10, 10]

    # test_diagonal_joins: 26-connectivity joins corner-touching voxels
    def test_diagonal_joins_corner_voxels(self) -> None:
        volume = np.zeros((4, 4, 4), dtype=np.int8)
        volume[0, 0, 0] = 1
        volume[1, 1, 1] = 1
        assert len(extract_components(volume)) == 1

    # test_empty: no foreground gives no components
    def test_all_zero_gives_no_components(self) -> None:
        assert extract_components(np.zeros((4, 4, 4), dtype=np.int8)) == []

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
