import pytest
import pandas as pd
import jsonlines
import numpy as np

from croissant.ingest import (
    parse_annotation, annotation_df_from_file, _ingest_uri, _keypath)


def generate_record(experiment_id, roi_id, label, annotation_labels=None):
    record = {
        "experiment-id": experiment_id,
        "roi-id": roi_id,
        "source-ref": "s3://bucket/input.png",
        "another-source-ref": "s3://bucket/input_1.png",
        "project": {
            "sourceData": "s3://bucket/input.png",
            "label": label,
        },
        "2-line-2-project-metadata": {
            "type": "groundtruth/custom",
            "job-name": "cool-job",
            "human-annotated": "yes",
            "creation-date": "2020-06-11T00:54:41.833000"
        }
    }
    if annotation_labels:
        worker_annotations = [{"workerId": "private-id", "roiLabel": annot}
                              for annot in annotation_labels]
        record["project"]["workerAnnotations"] = worker_annotations
    return record


record_3_worker = generate_record(867, 5309, "not cell",
                                  ["cell", "not cell", "not cell"])
record_no_worker = generate_record(000, 111, 0)


@pytest.mark.parametrize(
    "record", [record_3_worker],
)
def test_raises_error_on_missing(record):
    with pytest.raises(ValueError):
        parse_annotation(record, "project", "label", "workerAnnotations",
                         min_annotations=4, on_missing="error")


@pytest.mark.parametrize(
    "record, min_annotations, annotations_key, expected",
    [
        (record_3_worker, 3, "workerAnnotations", "not cell",),
        (record_3_worker, 1, "workerAnnotations", "not cell", ),
        (record_no_worker, 1, None, 0,),
    ]
)
def test_parse_annotation(record, min_annotations,
                          annotations_key, expected):
    # Mocking s3_get_object and the read method from the response
    label = parse_annotation(record, "project", "label",
                             annotations_key=annotations_key,
                             min_annotations=min_annotations,
                             on_missing="skip")
    assert label == expected


@pytest.mark.parametrize(
    "records, additional_keys, expected",
    [
        (
            [record_no_worker, record_3_worker],
            [("roi-id",), ("2-line-2-project-metadata", "job-name",)],
            pd.DataFrame({"roi-id": [111, 5309],
                          "job-name": ["cool-job", "cool-job"],
                          "label": [0, "not cell"]})
        ),
        (
            [record_no_worker, record_3_worker],
            None,
            pd.DataFrame({"label": [0, "not cell"]})
        ),
        (
            [record_no_worker, record_3_worker],
            [],
            pd.DataFrame({"label": [0, "not cell"]})
        )
    ]
)
def test_annotation_df_from_file(monkeypatch, tmp_path, records,
                                 additional_keys, expected):
    """Testing additional keys behavior, file loading and parser
    already unit tested.
    """
    monkeypatch.setattr("croissant.ingest.read_jsonlines",
                        lambda x: jsonlines.Reader(records, loads=lambda y: y))
    actual = annotation_df_from_file(
        records, "project", "label",
        annotations_key=None, min_annotations=1, on_missing="skip",
        additional_keys=additional_keys)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


@pytest.mark.parametrize(
    "keys, expected", [
        ("key1,key2", ("key1", "key2")),
        ("key1", ("key1",)),
    ]
)
def test_keypath(keys, expected):
    """Test that keypaths are parsed correctly from args"""
    assert expected == _keypath(keys)


@pytest.mark.parametrize(
    "roi_id_key, annotations_key, min_annotations, drop_na, expected",
    [
        (
            ("roi-id",), ("project", "workerAnnotations"), 3, False,
            pd.DataFrame(
                {"label": [1, 0], "roi_id": [456, 888],
                 "annotations": [[1, 0, 1], [0, 0, 0, 0]]})
        ),
        (   # Don't include additional keys
            None, None, 1, False,
            pd.DataFrame({"label": [1, 0]})
        ),
        (   # Bad data, but don't drop
            ("roi-id",), ("project", "workerAnnotations"), 4, False,
            pd.DataFrame(
                {"label": [np.NaN, 0], "roi_id": [456, 888],
                 "annotations": [[1, 0, 1], [0, 0, 0, 0]]})
        ),
        (   # Bad data, do drop
            ("roi-id",), ("project", "workerAnnotations"), 4, True,
            pd.DataFrame(
                {"label": [0], "roi_id": [888],
                 "annotations": [[0, 0, 0, 0]]}, index=[1])
        ),
    ]
)
def test_ingest_uri(monkeypatch, roi_id_key, annotations_key, min_annotations,
                    drop_na, expected):
    """Test the main module runner."""
    records = [
        generate_record(123, 456, "cell", ["cell", "not cell", "cell"]),
        generate_record(999, 888, "not cell",
                        ["not cell", "not cell", "not cell", "not cell"])]
    monkeypatch.setattr("croissant.ingest.read_jsonlines",
                        lambda x: jsonlines.Reader(records, loads=lambda y: y))
    actual = _ingest_uri(
        records, "project", "label", roi_id_key, annotations_key,
        min_annotations, "skip", drop_na)
    pd.testing.assert_frame_equal(expected, actual, check_like=True,
                                  check_dtype=False)
