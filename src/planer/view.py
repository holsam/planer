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

# SEGMENTATION_COLOURS: single neutral colour for the complete segmentation
SEGMENTATION_COLOURS = {
    None: (0.0, 0.0, 0.0, 0.0),
    0: (0.0, 0.0, 0.0, 0.0),
    1: (0.5, 0.5, 0.5, 1.0),
}

# SEVERITY_RGBA: red strictest, orange medium, yellow loosest
SEVERITY_RGBA = {
    Severity.LOW: (0.9, 0.1, 0.1, 1.0),
    Severity.MEDIUM: (1.0, 0.55, 0.0, 1.0),
    Severity.HIGH: (1.0, 0.9, 0.0, 1.0),
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
        _paint(shape, scored, np.ones(len(scored), dtype=np.uint8)),
        name='segmentation',
        colormap=DirectLabelColormap(color_dict=SEGMENTATION_COLOURS),
    )
    for severity in (Severity.LOW, Severity.MEDIUM, Severity.HIGH):
        viewer.add_labels(
            _paint(shape, scored, (confidences >= THRESHOLDS[severity]).astype(np.uint8)),
            name=f'removed: {severity.value}',
            colormap=DirectLabelColormap(
                color_dict={None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0), 1: SEVERITY_RGBA[severity]}
            ),
            visible=False,
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


# _paint: draw per-component values into a fresh volume
def _paint(shape: tuple[int, ...], scored: list[ScoredComponent], values: np.ndarray) -> np.ndarray:
    painted = np.zeros(shape, dtype=np.uint8)
    for item, value in zip(scored, values):
        painted[tuple(item.component.coords.T)] = value
    return painted
