'''
planer: remove bounding box artefacts from tomograms
'''

# Import external dependencies
from pathlib import Path

# Import internal planer objects
from planer.utils.io import read_mrc, write_mrc, write_scores
from planer.utils.judgement import judge
from planer.utils.schema import ScoredComponent, Severity, TrimReport, Verdict
from planer.utils.scoring import score_volume

# trim_volume: remove flagged components from a volume and write the cleaned volume plus scores
def trim_volume(
    source: Path,
    output_dir: Path,
    severity: Severity | float,
    *,
    overwrite: bool = False,
) -> TrimReport:
    output = output_dir / f'{source.stem}_trimmed.mrc'
    scores_path = output_dir / f'{source.stem}_scores.csv'
    # fail before the expensive scoring step
    if not overwrite:
        for target in (output, scores_path):
            if target.exists():
                raise FileExistsError(f'{target!r} exists, pass overwrite to replace it')
    output_dir.mkdir(parents=True, exist_ok=True)
    volume, voxel_size = read_mrc(source)
    scored = score_volume(volume)
    verdicts = [judge(item.scores, severity) for item in scored]
    removed = 0
    removed_voxels = 0
    for item, verdict in zip(scored, verdicts):
        if verdict.flagged:
            volume[tuple(item.component.coords.T)] = 0
            removed += 1
            removed_voxels += len(item.component.coords)
    write_mrc(output, volume, voxel_size, overwrite=overwrite)
    write_scores(scores_path, scored, verdicts)
    return TrimReport(
        source=source,
        output=output,
        scores=scores_path,
        total=len(scored),
        removed=removed,
        removed_voxels=removed_voxels,
    )

# trim_batch: trim several tomograms with shared settings
def trim_batch(
    sources: list[Path],
    output_dir: Path,
    severity: Severity | float,
    *,
    overwrite: bool = False,
) -> list[TrimReport]:
    return [trim_volume(source, output_dir, severity, overwrite=overwrite) for source in sources]
