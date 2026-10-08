'''
planer: unit tests for view command helpers
'''

# Import external dependencies
import numpy as np, pytest

# Import internal planer objects
napari = pytest.importorskip('napari')
from planer.utils.schema import Component, ScoredComponent
from planer.view import _paint

# TestView: unit tests for src/planer/view.py
class TestView:
    # test_paint_values: check each component is painted with its own value and the rest stays zero
    def test_paint_values(self) -> None:
        first = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32)
        second = np.array([[3, 3, 3]], dtype=np.int32)
        # scores unused by _paint
        scored = [ScoredComponent(component=Component(label=i, coords=c), scores=None) for i, c in enumerate([first, second], 1)]
        painted = _paint((4, 4, 4), scored, np.array([1, 0], dtype=np.uint8))
        assert painted[tuple(first.T)].tolist() == [1, 1]
        assert painted[tuple(second.T)].tolist() == [0]
        assert painted.sum() == 2
