'''
planer: unit tests for utility functions
'''

# Import external dependencies
import numpy as np, pytest
from pathlib import Path

# Import internal planer objects
from planer.utils.components import extract_components
from planer.utils.io import read_mrc, write_mrc
from planer.utils.judgement import THRESHOLDS, combine, judge, resolve_threshold
from planer.utils.schema import Component, Scores, Severity
from planer.utils.scoring import score_component, score_volume

# SHAPE: example volume shape
SHAPE = (64, 64, 64)

# _component: build a component from a boolean mask
def _component(mask: np.ndarray) -> Component:
    return Component(label=1, coords=np.argwhere(mask).astype(np.int32))

# TestComponents: tests for src/planer/utils/components.py
class TestComponents:
    # test_separate_blobs_return_different_components: two disjointed regions give two components
    def test_separate_blobs_return_different_components(self) -> None:
        volume = np.zeros((20, 20, 20), dtype=np.int8)
        volume[2:4, 2:4, 2:4] = 1
        volume[10:13, 10:13, 10:13] = 1
        components = extract_components(volume)
        assert [len(c.coords) for c in components] == [8, 27]
        assert components[1].coords.min(axis=0).tolist() == [10, 10, 10]

    # test_diagonal_joins_corner_voxels: 26-connectivity joins corner-touching voxels
    def test_diagonal_joins_corner_voxels(self) -> None:
        volume = np.zeros((4, 4, 4), dtype=np.int8)
        volume[0, 0, 0] = 1
        volume[1, 1, 1] = 1
        assert len(extract_components(volume)) == 1

    # test_all_zero_gives_no_components: no foreground gives no components
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

# TestJudgement: tests for src/planer/utils/judgement.py
class TestJudgement:
    # test_all_scores_high_is_flagged: three high scores get flagged at each severity level
    def test_all_scores_high_is_flagged(self) -> None:
        scores = Scores(flatness=1.0, orientation=1.0, proximity=0.95)
        assert all(judge(scores, level).flagged for level in Severity)

    # test_one_score_disagrees_blocks_flag: a near-zero score blocks flagging even when the others are perfect
    def test_one_score_disagrees_blocks_flag(self) -> None:
        scores = Scores(flatness=1.0, orientation=1.0, proximity=0.01)
        assert not judge(scores, Severity.STRICT).flagged

    # test_flag_depends_on_severity: marginal components flag only at stricter severities
    def test_flag_depends_on_severity(self) -> None:
        scores = Scores(flatness=0.8, orientation=0.8, proximity=0.8)
        assert combine(scores) == pytest.approx(0.8)
        assert not judge(scores, Severity.LENIENT).flagged
        assert judge(scores, Severity.MODERATE).flagged
        assert judge(scores, Severity.STRICT).flagged

    # test_explicit_threshold_overrides_presets: a float cutoff overrides the presets
    def test_explicit_threshold_overrides_presets(self) -> None:
        scores = Scores(flatness=0.8, orientation=0.8, proximity=0.8)
        assert judge(scores, 0.79).flagged
        assert not judge(scores, 0.81).flagged
        assert resolve_threshold(Severity.MODERATE) == THRESHOLDS[Severity.MODERATE]
        with pytest.raises(ValueError):
            resolve_threshold(1.5)

# TestScoring: tests for src/planer/utils/scoring.py
class TestScoring:
    # test_edge_slab_all_score_high: a flat, axis-aligned slab on a face scores high
    def test_edge_slab_all_score_high(self) -> None:
        mask = np.zeros(SHAPE, dtype=bool)
        mask[10:50, 10:50, 1] = True
        scores = score_component(_component(mask), SHAPE)
        assert scores.flatness > 0.95
        assert scores.orientation > 0.95
        assert scores.proximity > 0.9

    # test_central_slab_proximity_score_low: a flat slab in the centre has a low proximity score
    def test_central_slab_proximity_score_low(self) -> None:
        mask = np.zeros(SHAPE, dtype=bool)
        mask[10:50, 10:50, 32] = True
        scores = score_component(_component(mask), SHAPE)
        assert scores.flatness > 0.95
        assert scores.proximity < 0.2
        assert scores.orientation > 0.95

    # test_tilted_slab_orientation_score_low: a slab tilted off-axis has a low orientation score
    def test_tilted_slab_orientation_score_low(self) -> None:
        grid = np.indices(SHAPE)
        mask = (np.abs(grid[0] + grid[2] - 40) < 1) & (grid[1] > 10) & (grid[1] < 50)
        scores = score_component(_component(mask), SHAPE)
        assert scores.orientation < 0.1
        assert scores.flatness > 0.95

    # test_non_slab_flatness_scores_low: a cube has a low flatness score
    def test_non_slab_flatness_scores_low(self) -> None:
        mask = np.zeros(SHAPE, dtype=bool)
        mask[2:12, 2:12, 2:12] = True
        scores = score_component(_component(mask), SHAPE)
        assert scores.flatness < 0.1
        assert scores.orientation > 0.95

    # test_too_few_voxels_skips_scoring: components below the voxel limit are skipped so score zero for all
    def test_too_few_voxels_skips_scoring(self) -> None:
        mask = np.zeros(SHAPE, dtype=bool)
        mask[0, 0:3, 0:3] = True
        scores = score_component(_component(mask), SHAPE)
        assert (scores.flatness, scores.orientation, scores.proximity) == (0.0, 0.0, 0.0)

    # test_score_volume: a slab and a blob in one volume are scored separately
    def test_score_volume(self) -> None:
        volume = np.zeros(SHAPE, dtype=np.int8)
        volume[10:50, 10:50, 1] = 1
        volume[30:38, 30:38, 30:38] = 1
        scored = score_volume(volume)
        assert len(scored) == 2
        assert scored[0].scores.flatness > 0.95
        assert scored[1].scores.flatness < 0.1
