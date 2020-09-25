import pytest
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import skew
from typing import Any
import pandas as pd

from croissant.features import FeatureExtractor as fx
from croissant.features import ignore_col


@pytest.fixture
def basic_data():
    return (
        [   # rois
            {"id": 867, "coo_rows": [0], "coo_cols": [1],
                "coo_data": [1], "image_shape": (2, 3)},
            {"id": 5309, "coo_rows": [1], "coo_cols": [0],
                "coo_data": [1], "image_shape": (2, 3)},
        ],
        [   # traces
            [1, 1, 1], [2, 2, 2],
        ],
        [   # metadata
            {"genotype": "you"},
            {"genotype": "you"},
        ]
    )


@pytest.fixture
def function_class():
    """Test fixture for _feat_fns_by_source"""
    class FunctionClass():
        def __init__(self):
            pass

        def _feat_fn_arg(self, arg: Any):
            pass

        @classmethod
        def _feat_cls_fn_arg(cls, arg: Any):
            pass

        @staticmethod
        def _feat_static_fn_arg(arg: Any):
            pass

        def _feat_multi_arg(self, arg: Any, arg1: Any, *, arg2: Any = None):
            pass

        @staticmethod
        def _feat_no_arg():
            pass

        def _feat_different_arg(self, arg1: Any = None):
            pass

        def _feat_no_arg_self(self):
            pass

        def non_feat_arg(self, arg: Any):
            pass

    return FunctionClass


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.array([[1, 1, 1],
                              [0, 1, 0]])), 3/2),   # row long axis
        # col long axis, unit dimension
        (coo_matrix(np.array([[1, 0, 0],
                              [1, 0, 0]])), 2),
        (coo_matrix(np.array([[0, 1, 1],
                              [0, 0, 1]])), 1),    # equal axes
    ]
)
def test_ellipticalness(roi, expected):
    assert fx._feat_ellipticalness(roi) == expected


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.array([[0.1, 0.2, 0],
                              [0.9, 0, -1.]])), 4),
        (coo_matrix(np.array([[0.0],
                              [0.0]])), 0),
    ],
)
def test_area(roi, expected):
    assert fx._feat_area(roi) == expected


@pytest.mark.parametrize(
    "trace, expected",
    [
        (np.ones((10,)), 0),
        (np.arange(11), 0)      # Skew wouldn't be zero if length incorrect
    ]
)
def test_last_tenth_trace_skew(trace, expected):
    assert expected == fx._feat_last_tenth_trace_skew(trace)


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.ones((3, 6))), 3),
        (coo_matrix(np.ones((5,))), 1),
    ]

)
def test_roi_height(roi, expected):
    assert expected == fx._feat_roi_height(roi)


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.ones((3, 6))), 6),
        (coo_matrix(np.ones((5,))), 5),
    ]

)
def test_roi_width(roi, expected):
    assert expected == fx._feat_roi_width(roi)


@pytest.mark.parametrize(
    "roi, expected",
    [
        (
            coo_matrix(np.ones((3, 3))),
            (
                1, 0, 0,        # all of them have closest neighbor at r=1
                ((4*3) +        # 4 corners with 3 neighbors
                 (1*8) +        # central point with 8 neighbors
                 (4*5))/9,      # 4 middles with 5 neighbors
                np.std([3, 3, 3, 3, 8, 5, 5, 5, 5]),
                skew([3, 3, 3, 3, 8, 5, 5, 5, 5])
            ),
        ),
        (
            coo_matrix(([1], ([2], [2])), shape=(3, 3)), (0, 0, 0, 0, 0, 0)
        )
    ]
)
def test_neighborhood_info(roi, expected):
    actual = fx._neighborhood_info(roi)
    for ix, val in enumerate(expected):
        assert val == actual[ix]


@pytest.mark.parametrize(
    "rois, dff_traces, metadata, expected_ids, expected_rois",
    [
        (
            [   # all ids, Rois input
                {"id": 123, "coo_rows": [0], "coo_cols": [1],
                 "coo_data": [1], "image_shape": (2, 3)},
                {"id": 456, "coo_rows": [1], "coo_cols": [0],
                 "coo_data": [1], "image_shape": (2, 3)},
            ],
            [   # traces
                [1, 1, 1], [2, 2, 2],
            ],
            [   # metadata
                {"depth": 120},
                {"depth": 100},
            ],
            [123, 456],    # expected ids
            [   # expected rois data
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ]
        ),
        (
            [   # missing ids, Rois input
                {"id": 123, "coo_rows": [0], "coo_cols": [1],
                 "coo_data": [1], "image_shape": (2, 3)},
                {"coo_rows": [1], "coo_cols": [0],
                 "coo_data": [1], "image_shape": (2, 3)},
            ],
            [   # traces
                [1, 1, 1], [2, 2, 2],
            ],
            [   # metadata
                {"depth": 120},
                {"depth": 100},
            ],
            [],    # expected ids
            [   # expected rois data
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ]
        ),
        (
            [   # coo matrix ROIs
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ],
            [   # traces
                [1, 1, 1], [2, 2, 2],
            ],
            [   # metadata
                {"depth": 120},
                {"depth": 100},
            ],
            [],    # expected ids
            [   # expected rois data
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ]
        )
    ]
)
def test_FeatureExtractor_data_inputs(
        rois, dff_traces, metadata, expected_ids, expected_rois):
    """Test overloaded `rois` argument formatting."""
    extractor = fx(rois, dff_traces, metadata)
    assert extractor.roi_ids == expected_ids
    assert len(extractor.rois) == len(expected_rois)
    for e_roi, roi in zip(expected_rois, extractor.rois):
        np.testing.assert_equal(e_roi.toarray(), roi.toarray())


@pytest.mark.parametrize(
    "rois, traces, metadata",
    [
        (   # Unequal trace
            [coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
             coo_matrix(np.array([[0, 1, 0], [0, 0, 0]]))],
            [1],
            [{"rig": "N"}, {"rig": "S"}]
        ),
        (   # Unequal metadata
            [coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
             coo_matrix(np.array([[0, 1, 0], [0, 0, 0]]))],
            [1, 1],
            [{"rig": "N"}]
        ),
        (
            # Unequal ROI
            [coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
             coo_matrix(np.array([[0, 1, 0], [0, 0, 0]]))],
            [1, 1],
            [{"rig": "N"}]
        ),
    ]
)
def test_FeatureExtractor_nonequal_input_error(rois, traces, metadata):
    """Test unequal input lengths raise error."""
    with pytest.raises(ValueError):
        fx(rois, traces, metadata)


@pytest.mark.parametrize(
    "sources, expected",
    [
        (
            ["arg"],
            {
                "arg": ["_feat_fn_arg", "_feat_cls_fn_arg",
                        "_feat_static_fn_arg", "_feat_multi_arg"]
            }
        ),
        (
            ["arg", "arg1", "arg2"],
            {
                "arg": ["_feat_fn_arg", "_feat_cls_fn_arg",
                        "_feat_static_fn_arg", "_feat_multi_arg"],
                "arg1": ["_feat_different_arg", "_feat_multi_arg"],
                "arg2": []
            }
        )
    ]
)
def test_feat_fns_by_source(sources, expected, function_class):
    my_class = function_class()
    actual = fx._feat_fns_by_source(my_class, sources=sources)
    # Pytest doesn't have nice stuff for dicts... much less dicts with lists
    for k, vals in expected.items():
        assert k in actual
        assert len(actual[k]) == len(vals)
        for v in vals:
            assert v in actual[k]


@pytest.mark.parametrize(
    "xs, fns, expected",
    [
        ([1], (lambda x: "3a",), [("3a",)],),
        ([], (lambda x: 1,), [],),
        ([1, 2, 3], (lambda x: (x*2, 1), lambda x: x),
         [(2, 1, 1), (4, 1, 2), (6, 1, 3)],),
    ]
)
def test_apply_fns(xs, fns, expected):
    assert expected == fx._apply_functions(xs, *fns)


@pytest.fixture
def MockFeatureExtractor():
    class MockFeatureExtractor(fx):
        def __init__(self, *args):
            super().__init__(*args)

        @staticmethod
        def _testfeat_trace(trace):
            if trace[0] % 2 == 0:
                return "let"
            else:
                return "give"

        @staticmethod
        def _testfeat_roi1(roi):
            return "never"

        def _testfeat_roi2(self, roi):
            return "gonna"

    return MockFeatureExtractor


def test_feature_extractor_run(MockFeatureExtractor, monkeypatch, basic_data):
    mock_fx = MockFeatureExtractor(*basic_data)
    # Get the subset of functions defined in the mock class instead
    monkeypatch.setattr(
        mock_fx, "_feat_fns_by_source",
        lambda *args, **kwargs: {"roi": ["_testfeat_roi1", "_testfeat_roi2"],
                                 "trace": ["_testfeat_trace"]})
    monkeypatch.setattr(
        mock_fx, "_run_special_features",
        lambda: pd.DataFrame({"rig": ["up", "down"]}))
    expected = pd.DataFrame(
        [["never", "gonna", "give", "you", "up"],
         ["never", "gonna", "let", "you", "down"]],
        columns=["_testfeat_roi1", "_testfeat_roi2", "_testfeat_trace",
                 "genotype", "rig"],
        index=[867, 5309]
    )
    actual = mock_fx.run()
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "df, col_to_ignore, expected",
    [
        (pd.DataFrame({"abc": [1, 2, 3],
                       "difficulty": ["it's", "easy", "as"]}),
         ["abc"],
         pd.DataFrame({"difficulty": ["it's", "easy", "as"]}))
    ]
)
def test_ignore_col_ignores_existing_col(df, col_to_ignore, expected):
    actual = ignore_col(df, cols_to_ignore=col_to_ignore)
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "df, cols_to_ignore, expected",
    [
        (pd.DataFrame({"abc": [1, 2, 3],
                       "difficulty": ["it's", "easy", "as"],
                       "song": ["do", "re", "mi"]}),
         ["abc", "difficulty"],
         pd.DataFrame({"song": ["do", "re", "mi"]})),
    ]
)
def test_ignore_col_ignores_multiple_cols(df, cols_to_ignore, expected):
    actual = ignore_col(df, cols_to_ignore=cols_to_ignore)
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "df, cols_to_ignore, expected",
    [
        (pd.DataFrame({"abc": [1, 2, 3],
                       "difficulty": ["it's", "easy", "as"],
                       "song": ["do", "re", "mi"]}),
         ["abc", "michael"],
         pd.DataFrame({"difficulty": ["it's", "easy", "as"],
                       "song": ["do", "re", "mi"]})),
    ]
)
def test_ignore_col_raises_warning_nonexistent(df, cols_to_ignore, expected):
    with pytest.warns(UserWarning) as record:
        actual = ignore_col(df, cols_to_ignore=cols_to_ignore)

        assert len(record) == 1
        assert record[0].message.args[0] == (
            "Could not drop the following columns to ignore"
            "as they don't exist in the DataFrame: {'michael'}")
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "trace, expected",
    [
        (np.ones((10,)), 1.0),
        (np.array([1, 1, 1, 1, 3, 3, 3, 3]), 3/2),
    ]
)
def test_feat_max_to_avg_f_ratio(trace, expected):
    assert fx._feat_max_to_avg_f_ratio(trace) == expected
    assert fx._feat_max_to_avg_f_minus_np_ratio(trace) == expected


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.array([[0, 1, 0], [1, 1, 1]])), 4/6),
        (coo_matrix(np.array([1])), 1),
    ]
)
def test_feat_compactness(roi, expected):
    assert fx._feat_compactness(roi) == expected
