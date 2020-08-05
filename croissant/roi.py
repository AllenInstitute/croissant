from __future__ import annotations  # noqa
import sys
from typing import Optional, List, Tuple

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class Roi(TypedDict):
    """ROI mask image data.
    """
    roi_id: Optional[int]
    # Data for reconstructing mask on image planes
    coo_rows: List[int]
    coo_cols: List[int]
    coo_data: List[float]
    image_shape: Tuple[int, int]


class RoiMetadata(TypedDict):
    """Metadata associated experiments from which ROIs were extracted.
    """
    depth: int        # Microscope imaging depth
    full_genotype: str             # Mouse CRE line
    targeted_structure: str          # Targeted brain area (imaging)
    rig: str                  # Name of imaging rig


class RoiWithMetadata():
    """helper class for bundling together roi data and metadata
    and optionally a label and trace data
    """
    def __init__(self, roidata: Roi, metadata: RoiMetadata,
                 trace: Optional[List] = None,
                 label: Optional[bool] = None):
        self.trace = trace
        self.roi = roidata
        self.roi_meta = metadata
        self.label = label

    @classmethod
    def from_dict(cls, d: dict) -> RoiWithMetadata:  # noqa
        """constructs from a dictionary

        Parameters
        ----------
        d: dict
           required keys are those used for Roi(): roi_id, coo_cols, coo_rows,
           coo_data, image_shape and those used for RoiMetadata(): depth,
           full_genotype, targeted_structure, rig
           optional keys are: trace, label

        Returns
        -------
        RoiWithMetadata

        """
        roi = Roi(
                roi_id=d['roi_id'],
                coo_cols=d['coo_cols'],
                coo_rows=d['coo_rows'],
                coo_data=d['coo_data'],
                image_shape=d['image_shape'])
        roi_meta = RoiMetadata(
                depth=d['depth'],
                full_genotype=d['full_genotype'],
                targeted_structure=d['targeted_structure'],
                rig=d['rig'])
        return RoiWithMetadata(roi, roi_meta, d['trace'], d['label'])
