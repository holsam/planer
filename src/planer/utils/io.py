'''
planer: input/output utility functions
'''

# Import external dependencies
import csv, mrcfile, numpy as np
from pathlib import Path

# Import internal planer objects
from planer.utils.schema import ScoredComponent, Verdict, VoxelSize

# read_mrc: read volume from MRC file along with voxel size
def read_mrc(path: Path) -> tuple[np.ndarray, VoxelSize]:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data)
        size = mrc.voxel_size
        voxel_size = (float(size.x), float(size.y), float(size.z))
    if data.ndim != 3:
        raise ValueError(f'Expected a 3D volume, got shape {data.shape!r} from {path!r}')
    return data, voxel_size

# write_mrc: save a 3D array as an MRC file with the given voxel size
def write_mrc(
    path: Path,
    data: np.ndarray,
    voxel_size: VoxelSize,
    *,
    overwrite: bool = False,
) -> None:
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

# write_scores: save per-component scores and verdicts as csv file
def write_scores(path: Path, scored: list[ScoredComponent], verdicts: list[Verdict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label', 'voxels', 'flatness', 'orientation', 'proximity', 'confidence', 'flagged'])
        for item, verdict in zip(scored, verdicts):
            writer.writerow([
                item.component.label,
                len(item.component.coords),
                f'{item.scores.flatness:.4f}',
                f'{item.scores.orientation:.4f}',
                f'{item.scores.proximity:.4f}',
                f'{verdict.confidence:.4f}',
                int(verdict.flagged),
            ])
