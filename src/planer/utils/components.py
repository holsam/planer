'''
planer: connected component labelling
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label

# Import internal planer objects
from planer.utils.schema import Component

# CONNECTIVITY: 26-neighbour structure so diagonal contacts join components
CONNECTIVITY = generate_binary_structure(3, 3)

# extract_components: label foreground and keep per-component coordinates
def extract_components(volume: np.ndarray) -> list[Component]:
    labelled, _ = label(volume > 0, structure=CONNECTIVITY)
    components = []
    for index, region in enumerate(find_objects(labelled), start=1):
        if region is None:
            continue
        offset = np.array([axis.start for axis in region], dtype=np.int32)
        local = np.argwhere(labelled[region] == index).astype(np.int32)
        components.append(Component(label=index, coords=local + offset))
    return components
