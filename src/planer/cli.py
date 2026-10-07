'''
planer: command line interface for planer
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated

# Import internal planer objects
from planer.trim import trim_batch
from planer.utils.schema import Severity

# app: instantiate Typer class for planer
app = typer.Typer(
    add_completion=False,
    help='Detect and remove boundary artefacts from membrane segmentations.',
    no_args_is_help=True,
)

# trim_command: remove flagged artefact components from one or more MRCs
@app.command('trim')
def trim_command(
    inputs: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help='Path to one or more binary segmentation MRC files.')
    ],
    output_dir: Annotated[
        Path,
        typer.Option('--output-dir', '-o', help='Directory to write cleaned MRCs and score files to.')
    ] = Path('.'),
    severity: Annotated[
        Severity,
        typer.Option(help='Strictness preset to use for trimming.')
    ] = Severity.MEDIUM,
    threshold: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help='Explicit cutoff, overrides severity.')
    ] = None,
    *,
    overwrite: Annotated[
        bool,
        typer.Option(help='Overwrite existing output files.')
    ] = False,
) -> None:
    '''Remove bounding box segmentation artefacts from one or more volumes.'''
    cutoff = threshold if threshold is not None else severity
    try:
        reports = trim_batch(inputs, output_dir, cutoff, overwrite=overwrite)
    except FileExistsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1)
    for report in reports:
        typer.echo(f'{report.source.name}: removed {report.removed}/{report.total} components ({report.removed_voxels} voxels)')

# view_command: tune severity thresholds on one volume in napari
@app.command('view')
def view_command(
    volume: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help='Segmentation MRC.')
    ],
    cutoff: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help='Initial slider cutoff.')
    ] = 0.62,
) -> None:
    '''Visualise trimmed components at different thresholds.'''
    # napari is an optional extra, import only when needed
    try:
        from planer.view import launch_view
    except ImportError:
        typer.echo('planer view requires the dependency group "view" to be installed: uv tool install git+"https://github.com/holsam/planer[view]"', err=True)
        raise typer.Exit(code=1)
    launch_view(volume, cutoff=cutoff)
