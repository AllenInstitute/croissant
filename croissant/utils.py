"""Miscellaneous utility functions that don't quite belong elsewhere."""
import math
from functools import reduce
import inspect
import io
import json
import operator
import os.path
from typing import Any, Union, Generator, List
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
import h5py
from pathlib import Path
import jsonlines
import numpy as np
from scipy.signal import resample_poly


from croissant.roi import LimsRoiSchema


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


def json_write_local_or_s3(obj, uri: str):
    """Write a json to local storage or S3 path.

    Parameters
    ----------
    obj:
        Any json-writable object (compatible with `json.dump`).
    uri: str
        An S3 URI or path to local file system.
    """
    if uri.startswith("s3://"):
        parsed_s3 = urlparse(uri)
        bucket = parsed_s3.netloc
        file_key = parsed_s3.path[1:]
        s3 = boto3.client('s3')
        with io.BytesIO() as f:
            f.write(json.dumps(obj).encode())
            f.seek(0)
            s3.upload_fileobj(f, bucket, file_key)
    else:
        with open(uri, "w") as f:
            json.dump(obj, f)


def path_join_s3_os(path: str, to_join: str) -> str:
    """Analogous to os.path.join that works regardless of for s3
    or file system. The format doesn't matter for Unix machines since
    both the web and file system use '/', but does matter for Windows
    since paths use '\'.

    Parameters
    ----------
    path: str
        The base path
    to_join: str
        A string to join to the end of the base path with a system-
        appropriate character.

    Returns
    -------
    str
        The joined path.
    """
    if path.startswith("s3://"):
        return f"{path}/{to_join}"
    else:
        return os.path.join(path, to_join)


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


def _munge_traces(roi_data: List[LimsRoiSchema], trace_file_path: str,
                  trace_data_key: str, trace_names_key: str,
                  trace_sampling_rate: int, desired_trace_sampling_rate: int
                  ) -> List[np.ndarray]:
    """Read trace data from h5 file (in proper order) and downsample it.
    Parameters
    ----------
    roi_data : List[SparseAndDenseROI]
        A list of ROIs that conform to the SparseAndDenseROISchema.
    trace_file_path : str
        Path to h5 file containing trace data.
    trace_data_key : str
        Key used to access trace data.
    trace_names_key : str
        Key used to access the ROI name (id) corresponding to each trace.
    trace_sampling_rate : int
        Sampling rate (in Hz) of trace data.
    desired_trace_sampling_rate : int
        Desired sampling rate (in Hz) that trace data should be
        downsampled to.
    Returns
    -------
    List[np.1darray]
        A list of downsampled traces with each element corresponding to
        an element in roi_data.
    """
    traces_file = h5py.File(trace_file_path, "r")
    traces_data = traces_file[trace_data_key]

    # An array of str(int) describing roi names (id) associated with each trace
    # Example: ['10', '100', ..., '2', '20', '200', ..., '3']
    traces_id_order = traces_file[trace_names_key][:].astype(int)
    traces_id_mapping = {val: ind for ind, val in enumerate(traces_id_order)}

    downsampled_traces = []
    for roi in roi_data:
        assert roi['id'] == traces_id_order[traces_id_mapping[roi['id']]]
        trace_indx = traces_id_mapping[roi["id"]]
        ds_trace = downsample(traces_data[trace_indx, :],
                              trace_sampling_rate,
                              desired_trace_sampling_rate)
        downsampled_traces.append(ds_trace)

    traces_file.close()
    return downsampled_traces


def downsample(trace: np.ndarray, input_fps: int, output_fps: int):
    """Downsample 1d array using scipy resample_poly.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly    # noqa
    for more information.
    Parameters
    ----------
    trace: np.ndarray
        1d array of values to downsample
    input_fps: int
        The FPS that the trace data was captured at
    output_fps: int
        The desired FPS of the trace
    Returns
    -------
    np.ndarray
        1d array of values, downsampled to output_fps
    """
    if input_fps == output_fps:
        return trace
    elif output_fps > input_fps:
        raise ValueError("Output FPS can't be greater than input FPS.")
    gcd = math.gcd(input_fps, output_fps)
    up = output_fps / gcd
    down = input_fps / gcd
    downsample = resample_poly(trace, up, down, axis=0, padtype="median")
    return downsample
