'''
planer: interactive napari inspector for tuning severity thresholds
'''

# Import external dependencies
import napari, numpy as np
from magicgui import magicgui
from napari.utils.colormaps import DirectLabelColormap
from pathlib import Path

# Import internal planer objects
from planer.utils.io import read_mrc
from planer.utils.judgement import THRESHOLDS, combine
from planer.utils.schema import ScoredComponent, Severity
from planer.utils.scoring import score_volume

# LEVEL_KEPT: not removed at any severity
LEVEL_KEPT = 1

# LEVEL_HIGH: removed only at high severity (loosest threshold)
LEVEL_HIGH = 2

# LEVEL_MEDIUM: removed at medium severity and above
LEVEL_MEDIUM = 3

# LEVEL_LOW: removed at every severity including low (strictest threshold)
LEVEL_LOW = 4

# LEVEL_COLOURS: red loosest, orange medium, yellow strictest, grey kept
LEVEL_COLOURS = {
    None: (0.0, 0.0, 0.0, 0.0),
    0: (0.0, 0.0, 0.0, 0.0),
    LEVEL_KEPT: (0.5, 0.5, 0.5, 1.0),
    LEVEL_HIGH: (0.9, 0.1, 0.1, 1.0),
    LEVEL_MEDIUM: (1.0, 0.55, 0.0, 1.0),
    LEVEL_LOW: (1.0, 0.9, 0.0, 1.0),
}

# CUTOFF_COLOURS: highlight for components flagged at the slider cutoff
CUTOFF_COLOURS = {
    None: (0.0, 0.0, 0.0, 0.0),
    0: (0.0, 0.0, 0.0, 0.0),
    1: (0.0, 0.9, 0.9, 0.6),
}

# launch_view: open one volume in napari with live threshold controls
def launch_view(path: Path, *, cutoff: float = 0.62) -> None:
    volume, _ = read_mrc(path)
    scored = score_volume(volume)
    shape = tuple(int(n) for n in volume.shape)
    confidences = np.array([combine(item.scores) for item in scored])
    del volume  # only the sparse components are needed from here

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_labels(
        _paint(shape, scored, _levels(confidences)),
        name='severity levels',
        colormap=DirectLabelColormap(color_dict=LEVEL_COLOURS),
    )
    highlight = viewer.add_labels(
        _paint(shape, scored, (confidences >= cutoff).astype(np.uint8)),
        name='flagged at cutoff',
        colormap=DirectLabelColormap(color_dict=CUTOFF_COLOURS),
    )

    @magicgui(
        auto_call=True,
        cutoff={'widget_type': 'FloatSlider', 'min': 0.0, 'max': 1.0, 'step': 0.01},
        preset={'choices': ['custom', *[level.value for level in Severity]]},
    )
    def controls(cutoff: float = cutoff, preset: str = 'custom') -> None:
        threshold = cutoff if preset == 'custom' else THRESHOLDS[Severity(preset)]
        highlight.data = _paint(shape, scored, (confidences >= threshold).astype(np.uint8))

    viewer.window.add_dock_widget(controls, name='threshold')
    napari.run()


# _levels: map confidences to the strictest severity level that would remove each component
def _levels(confidences: np.ndarray) -> np.ndarray:
    return np.select(
        [
            confidences >= THRESHOLDS[Severity.LOW],
            confidences >= THRESHOLDS[Severity.MEDIUM],
            confidences >= THRESHOLDS[Severity.HIGH],
        ],
        [LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH],
        default=LEVEL_KEPT,
    ).astype(np.uint8)


# _paint: draw per-component values into a fresh volume
def _paint(shape: tuple[int, ...], scored: list[ScoredComponent], values: np.ndarray) -> np.ndarray:
    painted = np.zeros(shape, dtype=np.uint8)
    for item, value in zip(scored, values):
        painted[tuple(item.component.coords.T)] = value
    return painted
