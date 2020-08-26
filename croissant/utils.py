"""Miscellaneous utility functions that don't quite belong elsewhere."""
from typing import Any, Union, Generator
from functools import reduce
import operator
import boto3
from urllib.parse import urlparse
from pathlib import Path
import jsonlines
import json
import inspect
from botocore.exceptions import ClientError


def read_jsonlines(uri: Union[str, Path]) -> Generator[dict, None, None]:
    """
    Generator to load jsonlines file from either s3 or a local
    file, given a uri (s3 uri or local filepath).

    Parameters
    ----------
    uri: str or Path
        source file in jsonlines format

    Yields
    ------
    record: dict
        a single entry from the file

    """
    if str(uri).startswith("s3://"):
        data = s3_get_object(uri)["Body"].iter_lines(chunk_size=8192)    # The lines can be big        # noqa
        reader = jsonlines.Reader(data)
    else:
        data = open(uri, "rb")
        reader = jsonlines.Reader(data)
    for record in reader:
        yield record
    reader.close()


def nested_get_item(obj: dict, key_list: list) -> Any:
    """
    Get item from a list of keys, which define nested paths.

    Example:
    ```
    d = {"a": {"b": {"c": 1}}}
    keys = ["a", "b", "c"]
    nested_get_item(d, keys)    # 1
    ```
    Parameters
    ==========
    obj: dict
        The dictionary to retrieve items by key
    key_list: list
        List of keys, in order, to traverse through the nested dict
        and return a value.
    Raises
    ======
    KeyError if any key from `key_list` is not present in `obj`
    """
    # Handle single key just in case, since a string is also an iterator
    if isinstance(key_list, str):
        key_list = [key_list]
    else:
        key_list = list(key_list)
    if len(key_list) == 0:
        raise ValueError("Empty list is not a valid key.")
    return reduce(operator.getitem, key_list, obj)


def json_load_local_or_s3(uri: str) -> dict:
    """read a json from a local or s3 path

    Parameters
    ----------
    uri: str
        S3 URI or local filepath

    Returns
    -------
    jobj: object
        deserisalized ouput of json.load()

    """
    if uri.startswith("s3://"):
        fp = s3_get_object(uri)["Body"]
    else:
        fp = open(uri, "r")
    jobj = json.load(fp)
    fp.close()
    return jobj


def s3_get_object(uri: str) -> dict:
    """
    Utility wrapper for calling get_object from the boto3 s3 client,
    using an s3 URI directly (rather than having to parse the bucket
    and key)
    Parameters
    ==========
    uri: str
        Location of the s3 file object
    Returns
    =======
    Dict containing response from boto3 s3 client
    """
    s3 = boto3.client("s3")
    parsed_s3 = urlparse(uri)
    bucket = parsed_s3.netloc
    file_key = parsed_s3.path[1:]
    response = s3.get_object(Bucket=bucket, Key=file_key)
    return response


def object_exists(bucket, key):
    """whether an object exists in an S3 bucket
    Parameters
    ----------
    bucket: str
        name of bucket
    key: str
        object key. If not passed, object key will be
        basename of the file_name
    Returns
    -------
    exists : bool
    """
    exists = False
    client = boto3.client('s3')
    try:
        client.head_object(Bucket=bucket, Key=key)
        exists = True
    except ClientError:
        pass
    return exists


def is_prefixed_function_or_method(obj: Any, prefix: str = ""):
    """
    Returns true if the object is a method/function and the name
    of the object starts with `prefix`, otherwise returns False.
    """
    try:
        if ((inspect.ismethod(obj) or inspect.isfunction(obj)) and
                obj.__name__.startswith(prefix)):
            return True
    except AttributeError:
        pass
    return False
