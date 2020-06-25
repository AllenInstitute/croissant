import pytest
import pandas as pd
from moto import mock_s3
import boto3
import jsonlines

from croissant.ingest import (
    parse_annotation, annotation_df_from_file, _read_jsonlines)


record_no_worker = {
    "experiment-id": 000,
    "roi-id": 111,
    "source-ref": "s3://bucket/input.png",
    "another-source-ref": "s3://bucket/input_1.png",
    "project": {
        "sourceData": "s3://bucket/input.png",
        "label": 0,
    },
    "2-line-2-project-metadata": {
        "type": "groundtruth/custom",
        "job-name": "cool-job",
        "human-annotated": "yes",
        "creation-date": "2020-06-11T00:54:41.833000"
    }
}

record_3_worker = {
    "experiment-id": 867,
    "roi-id": 5309,
    "source-ref": "s3://bucket/input.png",
    "another-source-ref": "s3://bucket/input_1.png",
    "project": {
        "sourceData": "s3://bucket/input.png",
        "label": "not cell",
        "workerAnnotations": [
            {
                "workerId": "private.us-west-2.aa",
                "roiLabel": "not cell"
            },
            {
                "workerId": "private.us-west-2.bb",
                "roiLabel": "cell"
            },
            {
                "workerId": "private.us-west-2.cc",
                "roiLabel": "not cell"
            }
        ]
    },
    "2-line-2-project-metadata": {
        "type": "groundtruth/custom",
        "job-name": "cool-job",
        "human-annotated": "yes",
        "creation-date": "2020-06-11T00:54:41.833000"
    }
}


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
    monkeypatch.setattr("croissant.ingest._read_jsonlines",
                        lambda x: jsonlines.Reader(records, loads=lambda y: y))
    actual = annotation_df_from_file(
        records, "project", "label",
        annotations_key=None, min_annotations=1, on_missing="skip",
        additional_keys=additional_keys)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


@pytest.mark.parametrize(
    "body, expected",
    [
        (b'{"a": 1, "b": 3}\n{"b": 2}', [{"a": 1, "b": 3}, {"b": 2}]),
        (b'{"a": 1}', [{"a": 1}]),
        (b'', []),
    ]
)
def test_read_jsonlines_file(tmp_path, body, expected):
    with open(tmp_path / "filename", "wb") as f:
        f.write(body)
    reader = _read_jsonlines(tmp_path / "filename")
    response = []
    for record in reader:
        response.append(record)
    reader.close()
    assert expected == response


@mock_s3
@pytest.mark.parametrize(
    "body, expected",
    [
        (b'{"a": 1, "b": 3}\n{"b": 2}', [{"a": 1, "b": 3}, {"b": 2}]),
        (b'{"a": 1}', [{"a": 1}]),
        (b'', []),
    ]
)
def test_read_jsonlines_s3(body, expected):
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="mybucket")
    s3.put_object(Bucket="mybucket", Key="my/file.json",
                  Body=body)
    reader = _read_jsonlines("s3://mybucket/my/file.json")
    response = []
    for record in reader:
        response.append(record)
    reader.close()
    assert expected == response
