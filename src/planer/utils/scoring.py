'''
planer: per-component flatness, orientation and proximity scoring utilities
'''

# Import external dependencies
import numpy as np

# Import internal planer objects
from planer.utils.components import extract_components
from planer.utils.schema import Component, Pca, Scores, ScoredComponent

# MIN_VOXELS: minimum number of voxels needed within a component to run PCA
MIN_VOXELS = 30

# AXIS_FULL_DEGREES: angle from axis up to which component normals score 1
AXIS_FULL_DEGREES = 5.0

# AXIS_ZERO_DEGREES: angle from axis at which component normals score 0
AXIS_ZERO_DEGREES = 20.0

# PROXIMITY_SCALE: distance from face (in voxels) for extent of decaying proximity score
PROXIMITY_SCALE = 15.0

# compute_pca: PCA of voxel coordinates
def compute_pca(coords: np.ndarray) -> Pca:
    centred = coords.astype(np.float64) - coords.mean(axis=0)
    covariance = centred.T @ centred / len(coords)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending order
    return Pca(eigenvalues=eigenvalues, normal=eigenvectors[:, 0])

# flatness_score: compute score for componnet flatness (1 for a thin slab, 0 for a blob or a line)
def flatness_score(pca: Pca) -> float:
    smallest, middle, _ = pca.eigenvalues
    if middle <= 0:  # line or point
        return 0.0
    return float(np.clip(1 - smallest / middle, 0.0, 1.0))

# orientation_score: compute score for component orientation (1 when the normal is within set tolerance of a cardinal axis, decaying to 0 at set limit)
def orientation_score(pca: Pca) -> float:
    alignment = float(np.clip(np.max(np.abs(pca.normal)), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(alignment)))
    span = AXIS_ZERO_DEGREES - AXIS_FULL_DEGREES
    return float(np.clip(1 - (angle - AXIS_FULL_DEGREES) / span, 0.0, 1.0))

# proximity_score: calculate socre for component proximity (exponential closeness of the centroid to the nearest volume face
def proximity_score(coords: np.ndarray, shape: tuple[int, int, int]) -> float:
    centroid = coords.mean(axis=0)
    to_upper = np.array(shape) - 1 - centroid
    distance = max(float(min(centroid.min(), to_upper.min())), 0.0)
    return float(np.exp(-distance / PROXIMITY_SCALE))

# score_component: score one component from a given volume
def score_component(component: Component, shape: tuple[int, int, int]) -> Scores:
    if len(component.coords) < MIN_VOXELS:
        return Scores(flatness=0.0, orientation=0.0, proximity=0.0)
    pca = compute_pca(component.coords)
    return Scores(
        flatness=flatness_score(pca),
        orientation=orientation_score(pca),
        proximity=proximity_score(component.coords, shape),
    )

# score_volume: label a volume and score every component, shared by trim and inspect
def score_volume(volume: np.ndarray) -> list[ScoredComponent]:
    shape = tuple(int(n) for n in volume.shape)
    return [ScoredComponent(component=component, scores=score_component(component, shape)) for component in extract_components(volume)]
