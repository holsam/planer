# Planer

Detect and remove boundary arterfacts from membrane segmentations.

## Overview
Planer is a standalone tool for detecting and removing bounding box artefacts from binary segmentations of reconstructed cryoET volumes (e.g. those produced by [MemBrain-Seg](https://github.com/teamtomo/membrain-seg)).

### Methodology
Planer works by:
1. Splitting each segmentation volume to its constituent connected components (using 26-face connectivity).
2. Calculating normalised scores for flatness, orientation, and bounding box proximity.
    - Flatness score is derived from the ratio of the smallest eigenvalue from a PCA of the component's coordinates to the other two eigenvalues.
    - Orientation score is derived from the angle between the PCA normal (the eigenvector of the smallest eigenvalue) and the volume's cardinal axes.
    - Bounding box proximity is calculated as a distance-weighted closeness of the component's centroid to the nearest volume face.
3. Calculating an overall confidence value (the geometric mean of the three individual scores), and comparing this to pre- or user-defined thresholds to determine whether to include or exclude that component.

## Installation
```bash
# Install planer (without optional view dependencies)
uv tool install git+https://github.com/holsam/planer

# Install planer with optional viewer dependencies (allows visualisation of thresholds)
uv tool install git+"https://github.com/holsam/planer[view]"
```

## Commands
Planer features two commands: `trim` and `view`. 

### `planer trim`
`planer trim` accepts an MRC file (or multiple of these) and runs the planer workflow with the specified severity or threshold (default: medium severity). It then writes the cleaned MRC file to the specified output directory (default: current working directory).

```sh
planer trim [-o <output-directory>] [--severity low|medium|high] [--threshold <float>] <file.mrc>
```

### `planer view`
`planer view` accepts an MRC file, and runs the planer workflow at each severity level. It launches a napari window with the following layers:
- `segmentation`: the complete segmentation volume
- `removed: [high|medium|low]`: each layer shows the components removed at the respective severity level
- `removed: custom`: the components removed at a custom threshold set by the cutoff slider

```sh
planer view <file.mrc>
```
