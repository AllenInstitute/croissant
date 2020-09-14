from __future__ import annotations  # noqa
from typing import Union, List, Dict, Any, Callable, Iterable, Tuple, Optional
from functools import partial
import inspect
from itertools import chain
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KDTree
from scipy.sparse import coo_matrix
from scipy.stats import skew
from skimage.measure import moments_hu, moments_central, moments_normalized
import numpy as np
import pandas as pd

from croissant.roi import Roi, RoiMetadata, RoiWithMetadata
from croissant.utils import is_prefixed_function_or_method


class NeighborhoodInfo(object):
    """Bundle the KDTree queries"""
    def __init__(self, roi: coo_matrix):
        """
        Parameters
        ----------
        roi: coo_matrix
            An ROI in coo format
        """
        self.coords = list(zip(roi.col, roi.row))
        self.tree = KDTree(np.array(self.coords), leaf_size=10)

    def neighbor_dist(self, n: int = 1) -> Tuple[float]:
        """Compute statistics about n-nearest pixel neighbors (not
        including itself).
        Parameters
        ----------
        n: int
            The degree of distance from self. n=1 gets the nearest
            neighbor, n=2 gets the two nearest, etc.
        Returns
        -------
        Tuple[float]
            Tuple of mean, std, and skew of the distance to the `n`
            nearest pixel neighbors, for each pixel in the ROI
        """
        # Avoid ValueError if there's only one data point
        if len(self.coords) == 1:
            return (0, 0, 0)
        dist, _ = self.tree.query(self.coords, k=n+1)
        near_dist = [d[1] for d in dist]
        return np.mean(near_dist), np.std(near_dist), skew(near_dist)

    def adjacent_pixels(self) -> Tuple[float]:
        """Compute the number of adjacent pixels (including diagonal).
        Simple implementation, will 'double-count' when computing for
        adjacent pixels.
        Returns
        -------
        Tuple[float]
            Tuple of mean, std, and skew of the number of adjacent
            pixels (including diagonal) for each pixel in the ROI
        """
        ind = self.tree.query_radius(self.coords, r=1.42)
        vec_len = np.vectorize(len)
        n_neighbors = vec_len(ind) - 1    # Don't want itself as a neighbor
        return np.mean(n_neighbors), np.std(n_neighbors), skew(n_neighbors)


class FeatureExtractor:
    """Methods for extracting features from raw ROI data sources.
    """
    def __init__(self,
                 rois: Union[List[Roi], List[coo_matrix]],
                 f_traces: List[List[float]],
                 metadata: List[RoiMetadata],
                 np_traces: Optional[List[List[float]]] = None,
                 id_col: str = "id",
                 spike_stddev_threshold: float = 2.5):
        """
        Each index in rois, f_traces, and metadata should correspond
        to the same index in the other sources. That is, the roi mask
        in `rois[0]` should be associated with the trace in
        `f_traces[0]` and the metadata in `metadata[0]`. That also
        means that `rois`, `f_traces`, and `metadata` should all be
        the same length.

        It is optional to provide an ID for an ROI data point, if using
        the `Roi` input structure. All `Roi`s in the list of input data
        must have an ID (non null) otherwise this field will be ignored.

        To add feature methods to this class, it is important to follow
        the following convention:
            1. Prefix feature name with "_feat"
            2. Function signature must require only one positional
               argument, of the following values:
                   roi - a coo matrix, a single value of self.rois
                   trace - 1d array-like, a single value of self.traces
                   metadata - dict, a single value of self.metadata
            Functions that require additional arguments should be
            manually called during `run`, and not prefixed with `_feat`
            so they are not automatically collected. The appropriate
            arguments can be passed using instance attributes or
            arguments to the `run` function. These should have defaults.
        If this convention is not followed, the functions may not
        be automatically collected/executed properly when calling
        `FeatureExtractor.run`.

        Parameters
        ----------
        rois: Union[List[Roi], List[coo_matrix]]
            A list of ROI masks, where each mask is represented as
            either a coo_matrix, or a dictionary with the schema
            defined in `Roi`.
        f_traces: List[List[float]]
            A list of fluorescence traces for each ROI in `rois`.
        np_traces: List[List[float]]
            A list of neuropil fluorescence traces for each ROI in
            `rois`.
        metadata: List[RoiMetadata]
            A list of metadata dicts for each ROI in `rois`, with the
            schema described in `RoiMetadata`.
        id_col: str, default="id"
            Which column to use as as IDs for the ROI data (default="id").
            If not present, will not record IDs.

        spike_stddev_threshold: float, default=2.5
            Used as input to _simple_n_spikes to compute the number of
            spikes based on whether the data point in the trace exeeds
            the value of `trace_stddev_threshold * std(trace)`.
        Raises
        ------
        ValueError
            If the length of `rois`, `f_traces`, and `metadata`
            are not all equal.
        """
        if not (len(rois) == len(f_traces) == len(metadata)):
            raise ValueError("`rois`, `f_traces`, and `metadata` must be "
                             f"equal length. `rois`: {len(rois)}, "
                             f"`f_traces`: {len(f_traces)}, "
                             f"`metadata`: {len(metadata)}.")
        if (np_traces is not None) and (len(np_traces) != len(f_traces)):
            raise ValueError(("`f_traces` and `np_traces` must be "
                              f"equal length. `f_traces`: {len(f_traces)}, "
                              f"`np_traces`: {len(np_traces)}"))
        roi_ids = []
        coo_rois = []
        if isinstance(rois[0], dict):    # Convert Roi data to coo_matrix
            for roi in rois:
                id_ = roi.get(id_col)
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
        self.f_traces = f_traces        # ROI fluorescence traces
        self.np_traces = np_traces      # neuropil fluorescence traces
        self.f_minus_np_traces = None
        if self.np_traces is not None:
            self.f_minus_np_traces = (np.array(self.f_traces)
                                      - np.array(self.np_traces))
        self.roi_ids = roi_ids
        self.metadata = pd.DataFrame.from_records(metadata)

        # Feature-specific attributes
        self.spike_stddev_threshold = spike_stddev_threshold

    @classmethod
    def from_list_of_dict(
            self, data: List[Dict[str, Any]], id_col: str = None,
            spike_stddev_threshold: float = 2.5) -> FeatureExtractor:  # noqa
        """constructs FeatureExtractor from a list of dictionaries
        See FeatureExtractor.__init__ for explanation of additional
        parameters.

        Parameters
        ----------
        data: list of dict
            each dict conforms to format specified by
            RoiWithMetadata.from_dict()

        Returns
        -------
        FeatureExtractor

        """
        roi_list = [RoiWithMetadata.from_dict(r) for r in data]
        rois = [r.roi for r in roi_list]
        traces = [r.trace for r in roi_list]
        metas = [r.roi_meta for r in roi_list]
        if "np_trace" in data[0].keys():
            np_traces = [r.np_trace for r in roi_list]
        else:
            np_traces = None
        return FeatureExtractor(rois=rois, f_traces=traces, metadata=metas,
                                np_traces=np_traces, id_col=id_col,
                                spike_stddev_threshold=spike_stddev_threshold)

    @staticmethod
    def _feat_ellipticalness(roi: coo_matrix) -> float:
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
    def _feat_area(roi: coo_matrix) -> int:
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

    def _feat_simple_n_spikes(self, trace: np.ndarray) -> int:
        return self._simple_n_spikes(trace, self.spike_stddev_threshold)

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
        threshold = np.std(trace) * stddev_threshold
        return (trace > threshold).sum()

    @staticmethod
    def _feat_last_tenth_trace_skew(trace: np.ndarray) -> float:
        """Compute the skew for the last tenth of the data.
        This roughly corresponds to the 'fingerprint stimulus'
        at the end."""
        return skew(trace[-len(trace)//10:])

    @staticmethod
    def _feat_roi_width(roi: coo_matrix) -> int:
        """Maximum size of the ROI along the x axis."""
        return np.ptp(roi.col) + 1

    @staticmethod
    def _feat_roi_height(roi: coo_matrix) -> int:
        """Maximum size of the ROI along the y axis."""
        return np.ptp(roi.row) + 1

    @staticmethod
    def _feat_roi_data_std(roi: coo_matrix) -> float:
        """Standard deviation of the weighted mask data."""
        return np.std(roi.data)

    @staticmethod
    def _feat_roi_data_skew(roi: coo_matrix) -> float:
        """Skew of the weighted mask data."""
        return skew(roi.data)

    @staticmethod
    def _neighborhood_info(roi: coo_matrix) -> Tuple[float]:
        """Return information about pixel "neighbors". Since computing
        a kdtree can be expensive, return all these features at once
        instead of recomputing.

        Returns a tuple of the following information (Computed for each
        mask pixel and aggregated):
            average distance to nearest neighbor
            standard deviation of distance to nearest neighbor
            skew of distance to nearest neighbor
            average number of adjacent neighbors
            standard deviation of number of adjacent neighbors
            skew of number of adjacent neighbors
        """
        neighbor_extractor = NeighborhoodInfo(roi)
        dist_tuple = neighbor_extractor.neighbor_dist(1)
        adjacency_tuple = neighbor_extractor.adjacent_pixels()
        return tuple([*dist_tuple, *adjacency_tuple])

    @staticmethod
    def _feat_max_to_avg_f_ratio(trace: np.ndarray) -> float:
        """Return the ratio of the maximum value of the trace to the
        average value of the trace."""
        return np.nanmax(trace)/np.nanmean(trace)

    @staticmethod
    def _feat_max_to_avg_f_minus_np_ratio(f_minus_np: np.ndarray) -> float:
        """Return the ratio of the maximum value of the neuropil-subtracted
        trace to the average value of the neuropil-subtracted trace.
        Separated from _feat_max_to_avg_f_ratio for feature naming and
        automation."""
        return FeatureExtractor._feat_max_to_avg_f_ratio(f_minus_np)

    @staticmethod
    def _feat_compactness(roi: coo_matrix) -> float:
        """Returns the number of 'filled-in' pixels over the area of the
        bounding box."""
        pixels = FeatureExtractor._feat_area(roi)
        height = FeatureExtractor._feat_roi_height(roi)
        width = FeatureExtractor._feat_roi_width(roi)
        return pixels / (height * width)

    @staticmethod
    def _hu_moments(roi: coo_matrix) -> np.ndarray:
        """Returns the 7 Hu moments for an ROI image. See 
        https://scikit-image.org/docs/0.17.x/api/skimage.measure.html#moments-hu        # noqa
        for more information.

        Returns
        -------
        7-element, 1d np.array of Hu's image moments

        References
        ----------
        M. K. Hu, “Visual Pattern Recognition by Moment Invariants”, 
        IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962
        """
        roi_image = roi.toarray()
        mu = moments_central(roi_image)
        nu = moments_normalized(mu)
        return moments_hu(nu)

    def _run_special_features(self) -> pd.DataFrame:
        """
        Separate call for special features that don't play nice with the
        automated apply function, because they return more than one
        value. Kept separate from `run` to keep it cleaner, since it's
        the main entry point. Returns a dataframe of features.
        """
        # Add special features that don't play nice with automation
        neighbor_cols = ["avg_dist_nn", "std_dist_nn", "skew_dist_nn",
                         "avg_adjacent_n", "std_adjacent_n", "skew_adjacent_n"]
        hu_cols = [f"hu{i}" for i in range(7)]

        neighborhood_features = pd.DataFrame(self._apply_functions(
            self.rois, self._neighborhood_info), columns=neighbor_cols)
        hu_features = pd.DataFrame(self._apply_functions(
            self.rois, self._hu_moments), columns=hu_cols)
        all_features = pd.concat([neighborhood_features, hu_features], axis=1)
        return all_features

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
            A pandas dataframe of extracted features. The columns of the
            dataframe are the name of the functions (except for
            values from `metadata` -- those are the keys), and are in
            sorted order.
        """
        feature_fns_dict = self._feat_fns_by_source(
            self, sources=["roi", "trace", "np_trace", "f_minus_np"])
        roi_data = self._apply_functions(
            self.rois,
            *[getattr(self, call) for call in feature_fns_dict["roi"]])
        trace_data = self._apply_functions(
            self.f_traces,
            *[getattr(self, call) for call in feature_fns_dict["trace"]])
        feature_cols = feature_fns_dict["roi"] + feature_fns_dict["trace"]

        # Optional neuropil-subtracted trace features
        if self.f_minus_np_traces is not None:
            np_sub_data = self._apply_functions(
                self.f_minus_np_traces,
                *[getattr(self, call) for call in
                  feature_fns_dict["f_minus_np"]])
            data_iterable = zip(roi_data, trace_data, np_sub_data)
            feature_cols.extend(feature_fns_dict["f_minus_np"])
        else:
            data_iterable = zip(roi_data, trace_data)

        extracted_features = pd.DataFrame(
            list(tuple(chain.from_iterable(d)) for d in data_iterable),
            columns=feature_cols)

        # Feature callables that return more than one value and have
        # to be handled specially
        special_features = self._run_special_features()

        # Combine all features
        features = pd.concat(
            [extracted_features, self.metadata, special_features], axis=1)

        # Take sorted columns so that ordering is deterministic
        features = features[sorted(list(features))]
        if len(self.roi_ids):
            features.index = self.roi_ids
        return features

    @staticmethod
    def _feat_fns_by_source(
            obj: Any,
            sources: List[str] = ["roi", "trace", "metadata"]
            ) -> Dict[str, List[str]]:
        """
        This internal method is a helper function to get all the
        functions prefixed with "_feat_" in this class (FeatureExtractor),
        that also have one of the sources in `sources` as an argument.
        The purpose of this is to allow for dynamically collecting
        feature functions as they are added without having to call them
        manually, and apply the appropriate data to them.

        Note: This currently assumes that a feature function only uses
        one source, and that the source is passed as a positional
        argument.
        Parameters
        ----------
        obj: Any
            Technically any object but should be a class instance
        sources: List
            List of strings of data "sources". In this case we want to
            separate out the functions that use the rois vs. traces vs.
            metadata, etc. so that they can be properly called in
            batches.
        Returns
        -------
        dict
            A dictionary of source to function name that uses that source
            as an argument.
            Example:
                {"roi": ["_feat_roi_height", ...]
                 "trace": [...]}
        """
        is_feat = partial(is_prefixed_function_or_method, prefix="_feat_")
        feature_fns = inspect.getmembers(obj, is_feat)
        fn_dict = dict(zip(sources, [[] for x in sources]))
        for fn_name, fn in feature_fns:
            args = inspect.getfullargspec(fn).args
            for source in sources:
                if source in args:
                    fn_dict[source].append(fn_name)
        return fn_dict

    @staticmethod
    def _apply_functions(xs: Iterable, *fns: Callable) -> List[tuple]:
        """
        Apply arbitrary number of functions to each element in an iterable,
        and collect the results into a list of tuples (where each value
        in the tuple is the result of applying one of the functions in
        `fns` to the element in `xs`.)
        Parameters
        ----------
        xs: Iterable
            An iterable of input data, such as a numpy array
        fns: Callable
            Functions to apply to each member in the iterable `xs`.
            Should take a single argument and return a single value,
            or a tuple of values to be unpacked.
        Returns
        -------
        List of tuples; values of the tuples are from applying each
        function in `fns` to each element in the iterable `xs`.
        """
        records = []
        for y in xs:
            results = []
            for fn in fns:
                result = fn(y)
                # Don't want to unpack strings into characters...
                if isinstance(result, str):
                    results.append((result,))
                else:
                    try:
                        results.append(tuple(result))
                    except TypeError:
                        results.append((result,))
            records.append(tuple(chain.from_iterable(results)))

        return records


def ignore_col(df: pd.DataFrame, *, cols_to_ignore: List[str]) -> pd.DataFrame:
    """
    A wrapper around Pandas' DataFrame.drop method. Used to safely
    drop columns of a DataFrame and raise an warning if any of the
    specified columns are not in the DataFrame. This is used in
    feature_pipeline to get the returned pipeline to ignore certain columns.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe
    cols_to_ignore: List[str]
        A list of columns to drop from the DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame form the arguments, but without the columns
        specified in cols_to_ignore.
    """
    df_cols = df.columns.tolist()

    if not set(cols_to_ignore).issubset(df_cols):
        warnings.warn("Could not drop the following columns to ignore"
                      "as they don't exist in the DataFrame: "
                      f"{set(cols_to_ignore).difference(df_cols)}")
        cols_to_ignore = list(
            set(df_cols).intersection(cols_to_ignore)
        )
    return df.drop(columns=cols_to_ignore)


def feature_pipeline(drop_cols: List[str] = None) -> Pipeline:
    """
    Create Pipeline to process extracted features from FeatureExtractor.
    Should be kept in sync with the supported features in
    FeatureExtractor.

    One-hot-encodes categorical features:
        full_genotype
        targeted_structure
        rig
    Include all other columns as numerical features.

    Parameters
    ----------
    drop_cols: List[str]
        A list of columns for the pipeline to ignore when fitting.

    Returns
    -------
    Pipeline
        Unfitted pipeline to process features extracted from
        FeatureExtractor.
    """
    drop_cols = [] if drop_cols is None else drop_cols

    categorical_cols = ["full_genotype", "targeted_structure", "rig"]
    # Remove dropped columns from the list of categorical columns
    categorical_cols = list(
        set(categorical_cols).difference(
            set(categorical_cols).intersection(drop_cols)))

    column_transformer = ColumnTransformer(
        transformers=[
            ("onehot_cat", OneHotEncoder(drop="if_binary"), categorical_cols)
        ],
        remainder="passthrough")

    function_transformer = FunctionTransformer(
        func=ignore_col,
        kw_args={"cols_to_ignore": drop_cols}
    )
    steps = [("onehot", column_transformer)]
    if drop_cols:
        # Want to drop first if we're doing it
        steps.insert(0, ("drop_cols", function_transformer))
    feature_pipeline = Pipeline(steps=steps)
    return feature_pipeline
