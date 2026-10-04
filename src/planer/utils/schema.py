'''
planer: schema objects used in planer
'''

# Import external dependencies
from dataclasses import dataclass

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

# VoxelSize: voxel edge lengths in Å (x,y,z)
type VoxelSize = tuple[float, float, float]
