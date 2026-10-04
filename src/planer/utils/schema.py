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

# VoxelSize: voxel edge lengths in Å (x,y,z)
type VoxelSize = tuple[float, float, float]
