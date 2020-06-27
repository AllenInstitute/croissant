from typing import Union, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import coo_matrix
from scipy.stats import skew
import numpy as np
import pandas as pd

from croissant.roi import Roi, RoiMetadata, RoiWithMetadata


class FeatureExtractor:
    """Methods for extracting features from raw ROI data sources.
    """
    def __init__(self,
                 rois: Union[List[Roi], List[coo_matrix]],
                 dff_traces: List[List[float]],
                 metadata: List[RoiMetadata]):
        """
        Each index in rois, dff_traces, and metadata should correspond
        to the same index in the other sources. That is, the roi mask
        in `rois[0]` should be associated with the trace in
        `dff_traces[0]` and the metadata in `metadata[0]`. That also
        means that `rois`, `dff_traces`, and `metadata` should all be
        the same length.

        It is optional to provide an ID for an ROI data point, if using
        the `Roi` input structure. All `Roi`s in the list of input data
        must have an ID (non null) otherwise this field will be ignored.

        Parameters
        ----------
        rois: Union[List[Roi], List[coo_matrix]]
            A list of ROI masks, where each mask is represented as
            either a coo_matrix, or a dictionary with the schema
            defined in `Roi`.
        dff_traces: List[List[float]]
            A list of dff_traces for each ROI in `rois`.
        metadata: List[RoiMetadata]
            A list of metadata dicts for each ROI in `rois`, with the
            schema described in `RoiMetadata`.

        Raises
        ------
        ValueError
            If the length of `rois`, `dff_traces`, and `metadata`
            are not all equal.
        """
        if not (len(rois) == len(dff_traces) == len(metadata)):
            raise ValueError("`rois`, `dff_traces`, and `metadata` must be "
                             f"equal length. `rois`: {len(rois)}, "
                             f"`dff_traces`: {len(dff_traces)}, "
                             f"`metadata`: {len(metadata)}.")
        roi_ids = []
        coo_rois = []
        if isinstance(rois[0], dict):    # Convert Roi data to coo_matrix
            for roi in rois:
                id_ = roi.get("id")
                if id_:
                    roi_ids.append(id_)
                coo_rois.append(coo_matrix((roi["coo_data"],
                                (roi["coo_rows"], roi["coo_cols"])),
                                roi["image_shape"]))
            if len(roi_ids) != len(rois):
                roi_ids = []
        else:
            coo_rois = rois

        self.rois = coo_rois
        self.dff_traces = dff_traces
        self.roi_ids = roi_ids
        self.metadata = pd.DataFrame.from_records(metadata)

    @classmethod
    def from_list_of_dict(self, data):
        roi_list = [RoiWithMetadata.from_dict(r) for r in data]
        rois = [r.roi for r in roi_list]
        traces = [r.trace for r in roi_list]
        metas = [r.roi_meta for r in roi_list]
        return FeatureExtractor(rois=rois, dff_traces=traces, metadata=metas)

    @staticmethod
    def _ellipticalness(roi: coo_matrix) -> float:
        """
        Compute the 'ellipticalness' of a sparse matrix by dividing the
        length of the long axis by the length of the short axis.

        Parameters
        ----------
        roi: coo_matrix
            COO matrix representation of an ROI mask extracted from image
        Returns
        -------
        float
            'ellipticalness' of the roi mask shape

        """
        r_axis_len = roi.row.max() - roi.row.min() + 1
        c_axis_len = roi.col.max() - roi.col.min() + 1
        if r_axis_len > c_axis_len:
            return r_axis_len/c_axis_len
        else:
            return c_axis_len/r_axis_len

    @staticmethod
    def _area(roi: coo_matrix) -> int:
        """
        Compute the 'area' of a sparse matrix by counting the number of
        nonzero pixels.

        Parameters
        ----------
        roi: coo_matrix
            COO matrix representation of an ROI mask extracted from image
        Returns
        -------
        int
            Total number of nonzero pixels of the matrix (the 'area')
        """
        return len(roi.data)

    @staticmethod
    def _simple_n_spikes(trace: np.ndarray,
                         stddev_threshold: float) -> int:
        """
        Simple spike extraction, based on whether the data point in the
        trace is above a certain number of standard deviations.

        Parameters
        ----------
        trace: np.ndarray
            Fluorescence trace data (can be normalized, such as dF/F).
        stddev_threshold: float
            Standard deviation threshold to 'fence' the data. Data
            points that are greater than `std` * `stddev_threshold` will
            be be counted as spikes.
        Returns
        -------
        int
            Number of points where the data points exceeded the standard
            deviation threshold
        """
        threshold = trace.std() * stddev_threshold
        return (trace > threshold).sum()

    def run(self) -> pd.DataFrame:
        """
        Run feature extraction methods to create input data for ML
        model. Extracts features from ROI and trace data, and adds
        metadata features to the final dataframe.
        The result can be fed into `feature_pipeline` to produce
        a `Pipeline` object for further preprocessing the features
        (compatible with cross validation).

        The features produced by these methods do not rely on the
        distribution of the input data, (unlike e.g., mean removal
        and standard scaling), so they will not change depending on
        how the data are split during cross validation or otherwise.

        Returns
        -------
        pd.DataFrame:
            A pandas dataframe with the following features:
                'trace_skew': skew of trace
                'roi_area': area of ROI mask
                'roi_ellipticalness': 'ellipticalness' of ROI mask
                'targeted_structure: targeted brain region
                'rig': imaging rig name
                'depth': imaging depth (of plane)
                'full_genotype': CRE line of mouse
        """
        trace_skew = list(map(skew, self.dff_traces))
        roi_area = list(map(self._area, self.rois))
        roi_ellipticalness = list(map(self._ellipticalness, self.rois))
        extracted_features = pd.DataFrame(
            {
                "trace_skew": trace_skew,
                "roi_area": roi_area,
                "roi_ellipticalness": roi_ellipticalness
            }
        )
        features = pd.concat([extracted_features, self.metadata], axis=1)
        if len(self.roi_ids):
            features.index = self.roi_ids
        return features


def feature_pipeline() -> Pipeline:
    """
    Create Pipeline to process extracted features from FeatureExtractor.
    Should be kept in sync with the supported features in
    FeatureExtractor.

    One-hot-encodes categorical features:
        full_genotype
        targeted_structure
        rig
    Include all other columns as numerical features.

    Returns
    -------
    Pipeline
        Unfitted pipeline to process features extracted from
        FeatureExtractor.
    """
    categorical_cols = ["full_genotype", "targeted_structure", "rig"]
    column_transformer = ColumnTransformer(
        transformers=[
            ("onehot_cat", OneHotEncoder(drop="if_binary"), categorical_cols)
        ],
        remainder="passthrough")
    feature_pipeline = Pipeline(
        steps=[("onehot", column_transformer)])
    return feature_pipeline
