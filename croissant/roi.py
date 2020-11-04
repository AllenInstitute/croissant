from __future__ import annotations  # noqa
import sys
from typing import Optional, List, Tuple
import warnings

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from marshmallow import Schema, fields, exceptions, post_load
import numpy as np
from scipy.sparse import coo_matrix


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


class LimsRoiSchema(Schema):
    """This ROI format is the expected output of Segmentation/Binarization.
    Copied from ophys_etl to avoid circular dependency (I know...)
    """

    id = fields.Int(
        required=True,
        description=("Unique ID of the ROI, get's overwritten writting "
                     "to LIMS"))
    x = fields.Int(
        required=True,
        description="X location of top left corner of ROI in pixels")
    y = fields.Int(
        required=True,
        description="Y location of top left corner of ROI in pixels")
    width = fields.Int(
        required=True,
        description="Width of the ROI in pixels")
    height = fields.Int(
        required=True,
        description="Height of the ROI in pixels")
    valid_roi = fields.Bool(
        required=True,
        description=("Boolean indicating if the ROI is a valid "
                     "cell or not"))
    mask_matrix = fields.List(
        fields.List(fields.Bool),
        required=True,
        description=("Bool nested list describing which pixels "
                     "in the ROI area are part of the cell"))
    max_correction_up = fields.Float(
        required=True,
        description=("Max correction in pixels in the up direction"))
    max_correction_down = fields.Float(
        required=True,
        description=("Max correction in pixels in the "
                     "down direction"))
    max_correction_left = fields.Float(
        required=True,
        description=("Max correction in pixels in the "
                     "left direction"))
    max_correction_right = fields.Float(
        required=True,
        description="Max correction in the pixels in the right direction")
    mask_image_plane = fields.Int(
        required=True,
        description=("The old segmentation pipeline stored "
                     "overlapping ROIs on separate image "
                     "planes. For compatibility purposes, "
                     "this field must be kept, but will "
                     "always be set to zero for the new "
                     "updated pipeline"))
    exclusion_labels = fields.List(
        fields.Str,
        required=True,
        description=("LIMS ExclusionLabel names used to "
                     "track why a given ROI is not "
                     "considered a valid_roi. (examples: "
                     "motion_border, "
                     "classified_as_not_cell)"))

    @post_load
    def add_coo_data(self, data, **kwargs):
        """Convert ROIs to coo format, which is used by the croissant
        FeatureExtractor. Input includes 'x' and 'y' fields
        which designate the cartesian coordinates of the top right corner,
        the width and height of the bounding box, and boolean values for
        whether the mask pixel is contained. The returned coo_matrix
        will contain all the data in the mask in the proper shape,
        but essentially discards the 'x' and 'y' information (the
        cartesian position of the masks is not important for the
        below methods). Represented as a dense array, the mask data
        would be "cropped" to the bounding box.
        Note: If the methods were updated such that the position of
        the mask relative to the input data *were*
        important (say, if necessary to align the masks to the movie
        from which they were created), then this function would require
        the dimensions of the source movie.
        """
        shape = (data["height"], data["width"])
        arr = np.array(data["mask_matrix"]).astype("int")
        if data["height"] + data["width"] == 0:
            warnings.warn("Input data contains empty ROI. "
                          "This may cause problems.")
        elif arr.shape != shape:
            raise exceptions.ValidationError(
                "Data in mask matrix did not correspond to "
                "the (height, width) dimensions. Please "
                "check the input data.")
        mat = coo_matrix(arr)
        data.update({"coo_roi": mat})
        return data
