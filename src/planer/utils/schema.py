'''
planer: schema objects used in planer
'''

# Import external dependencies
from dataclasses import dataclass
from enum import StrEnum

# Component: one connected foreground region as voxel indices in volume space
@dataclass(frozen=True)
class Component:
    label: int
    coords: np.ndarray  # (n, 3) int32, never mutated

# Pca: eigen-decomposition of a component's coordinate covariance
@dataclass(frozen=True)
class Pca:
    eigenvalues: np.ndarray  # ascending, shape (3,)
    normal: np.ndarray  # unit eigenvector of the smallest eigenvalue

# Scores: the three normalised 0-1 scores for a component
@dataclass(frozen=True)
class Scores:
    flatness: float
    orientation: float
    proximity: float

# ScoredComponent: a component and its resulting scores
@dataclass(frozen=True)
class ScoredComponent:
    component: Component
    scores: Scores

# Severity: levels for how aggressively boundary box artefacts are flagged
class Severity(StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'

# Verdict: whether a component is flagged for removal, and how confident the judgment is
@dataclass(frozen=True)
class Verdict:
    flagged: bool
    confidence: float

# TrimReport: outcome of trimming one tomogram
@dataclass(frozen=True)
class TrimReport:
    source: Path
    output: Path
    scores: Path
    total: int
    removed: int
    removed_voxels: int

# VoxelSize: voxel edge lengths in Å (x,y,z)
type VoxelSize = tuple[float, float, float]
