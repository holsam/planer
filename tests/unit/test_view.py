'''
planer: unit tests for view command helpers
'''

# Import external dependencies
import numpy as np, pytest

# Import internal planer objects
napari = pytest.importorskip('napari')
from planer.view import LEVEL_HIGH, LEVEL_KEPT, LEVEL_LOW, LEVEL_MEDIUM, _levels

# TestView: unit tests for src/planer/view.py 
class TestView:
    # test_levels_map: check confidences map to the strictest removing level
    def test_levels_map(self) -> None:
        levels = _levels(np.array([0.9, 0.75, 0.6, 0.1]))
        assert levels.tolist() == [LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH, LEVEL_KEPT]
