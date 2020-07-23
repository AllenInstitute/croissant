import pytest
import boto3
from moto import mock_s3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import json

from croissant.utils import (nested_get_item, s3_get_object,
                             read_jsonlines, json_load_local_or_s3,
                             object_exists)


@pytest.mark.parametrize(
    "d, keys, expected",
    [
        ({"a": {1: {"c": "a"}}}, ["a", 1], {"c": "a"},),
        ({"a": {1: {"c": "a"}}}, ["a", 1, "c"], "a",),
        ({"a": "b"}, ["a"], "b",),
        ({"a": "b"}, "a", "b",),
    ]
)
def test_nested_get_item(d, keys, expected):
    assert expected == nested_get_item(d, keys)


@pytest.mark.parametrize(
    "d, keys",
    [
        ({"a": {1: {"c": "a"}}}, ["b", 2],),
        ({"a": {1: {"c": "a"}}}, ["a", 1, "s"],),
        ({"a": "b"}, ["s"],),
        ({"a": "b"}, "s",),
        ({}, "a"),
    ]
)
def test_nested_get_item_fails_if_missing_key(d, keys):
    with pytest.raises(KeyError):
        nested_get_item(d, keys)


@pytest.mark.parametrize(
    "d, keys",
    [
        ({"a": 1}, [],),
        ({}, [],),
    ]
)
def test_nested_get_item_fails_with_empty_list(d, keys):
    with pytest.raises(ValueError):
        nested_get_item(d, keys)


@mock_s3
def test_s3_get_object():
    # Set up the fake bucket and object
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="mybucket")
    body = b'{"a": 1}\n{"b": 2}'
    s3.put_object(Bucket="mybucket", Key="my/file.json",
                  Body=body)
    # Run the test
    response = s3_get_object("s3://mybucket/my/file.json")
    assert body == response["Body"].read()


@mock_s3
def test_s3_fails_not_exist():
    # Set up the fake bucket and object
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="mybucket")
    body = b'{"a": 1}\n{"b": 2}'
    s3.put_object(Bucket="mybucket", Key="my/file.json",
                  Body=body)
    # Run the test
    with pytest.raises(ClientError) as e:
        s3_get_object("s3://mybucket/my/nonexistentfile.json")
        assert e.response["Error"]["Code"] == "NoSuchKey"


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
    reader = read_jsonlines(tmp_path / "filename")
    response = []
    for record in reader:
        response.append(record)
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
    reader = read_jsonlines("s3://mybucket/my/file.json")
    response = []
    for record in reader:
        response.append(record)
    assert expected == response


@mock_s3
@pytest.mark.parametrize("expected", [{'test': 123}])
@pytest.mark.parametrize("mode", ["local", "s3"])
def test_json_load_local_or_s3(mode, expected, tmp_path):
    if mode == "s3":
        uri = "s3://myjsonbucket/my/file.json"
        up = urlparse(uri)
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=up.netloc)
        s3.put_object(Bucket=up.netloc, Key=up.path[1:],
                      Body=json.dumps(expected).encode('utf-8'))
    else:
        uri = str(tmp_path / "myfile.json")
        with open(uri, "w") as f:
            json.dump(expected, f)

    loaded = json_load_local_or_s3(uri)
    assert loaded == expected


@mock_s3
def test_object_exists():
    uri = "s3://myobjectbucket/my/file.json"
    up = urlparse(uri)
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket=up.netloc)
    s3.put_object(Bucket=up.netloc, Key=up.path[1:],
                  Body=json.dumps({'a': 1}).encode('utf-8'))

    assert object_exists(up.netloc, up.path[1:])
    assert not object_exists(up.netloc, "does/not/exist.txt")
