from typing import Optional, Any, Union, List, Tuple
from pathlib import Path
import pandas as pd
from functools import partial
import jsonlines

from croissant.utils import s3_get_object, nested_get_item


def _read_jsonlines(uri: Union[str, Path]) -> jsonlines.Reader:
    """
    Helper function to load jsonlines file from either s3 or a local
    file, given a uri (s3 uri or local filepath).
    """
    if str(uri).startswith("s3://"):
        data = s3_get_object(uri)["Body"].iter_lines(chunk_size=8192)    # The lines can be big        # noqa
        reader = jsonlines.Reader(data)
    else:
        data = open(uri, "rb")
        reader = jsonlines.Reader(data)
    return reader


def annotation_df_from_file(
        filepath: Union[str, Path],
        project_key: str,
        label_key: str,
        annotations_key: Optional[str] = None,
        min_annotations: int = 1,
        on_missing: str = "skip",
        additional_keys: List[Tuple] = None) -> pd.DataFrame:
    """
    Apply `parse_annotation` to a local file or a file stored in s3
    in jsonlines format, and return a dataframe of labels. Can also
    pass additional keys/key paths to extract from the source data
    and add as columns on the dataframe. The final key in each tuple
    in `additional_keys` will be used as the column name for those
    values in the returned dataframe.

    See `parse_annotation` for more information about enforcing
    `min_annotations` requirements.

    Parameters
    ==========
    filepath: str or pathlib.Path
        Path to a local file or an s3 storage location (prefixed by
        's3://'), in jsonlines format
    project_key: str
        The key that contains the dict of annotation data
    label_key: str
        The key that contains the final labeled value to extract,
        in the object (dict) indexed by `project_key`.
    annotations_key: str
        If applicable, the key in the record that contains the list of
        individual worker annotations (as dict records). Only used if
        `min_annotations` > 1.
    min_annotations: int
        Number of annotations required for a valid label. If > 1, checks
        the length of the annotation records in
        `record[label_key][annotations_key]`, which should be a list.
    on_missing: str
        One of "error", or "skip" (default="skip"). The function's
        behavior if a record is encountered that does not meet the
        `min_annotations` threshold (annotations are missing).
        Only used if `min_annotations` > 1.
        If "skip", function will return `None` for invalid records and
        log a warning. If "error", will raise a ValueError instead.
    additional_keys: list or List[list]
        List of tuples identifying sequential keys to access a value
        in the annotation record json using
        croissant.utils.nested_get_item. If the key is top-level, pass
        it as a 1-tuple.
    """
    reader = _read_jsonlines(filepath)
    parser = partial(parse_annotation, project_key=project_key,
                     label_key=label_key, annotations_key=annotations_key,
                     min_annotations=min_annotations, on_missing=on_missing)
    labels = {label_key: []}
    if additional_keys:
        getters = []
        # Get all keys so only traverse data once
        for keypath in additional_keys:
            labels.update({keypath[-1]: []})
            getters.append((keypath[-1], partial(nested_get_item,
                                                 key_list=keypath)))
        for record in reader:
            for getter in getters:
                labels[getter[0]].append(getter[1](record))
            labels[label_key].append(parser(record))
    else:
        labels = {label_key: list(map(parser, reader))}
    reader.close()
    return pd.DataFrame(labels)


def parse_annotation(record: dict,
                     project_key: str,
                     label_key: str,
                     annotations_key: Optional[str] = None,
                     min_annotations: int = 1,
                     on_missing: str = "skip") -> Any:
    """
    Parses an annotation record and returns the label value (from the
    `record[project_key][label_key]` key).
    Optionally enforces a requirement of `min_annotations` number of
    annotations. The format of the output manifest depends heavily on
    the post-annotation lambda functions. This assumes, minimally,
    that there is a dict of annotation data indexed by `project_key`.
    In this dict there is a key `label_key` that stores the value for
    the final label of this record.

    Parameters
    ==========
    record: dict
        A single record from output.manifest (output of SageMaker GT
        labeling job)
    project_key: str
        The key that contains the dict of annotation data
    label_key: str
        The key that contains the final labeled value to extract,
        in the object (dict) indexed by `project_key`.
    annotations_key: str
        If applicable, the key in the record that contains the list of
        individual worker annotations (as dict records). Only used if
        `min_annotations` > 1.
    min_annotations: int
        Number of annotations required for a valid label. If > 1, checks
        the length of the annotation records in
        `record[label_key][annotations_key]`, which should be a list.
    on_missing: str
        One of "error", or "skip" (default="skip"). The function's
        behavior if a record is encountered that does not meet the
        `min_annotations` threshold (annotations are missing).
        Only used if `min_annotations` > 1.
        If "skip", function will return `None` for invalid records and
        log a warning. If "error", will raise a ValueError instead.

    Returns
    =======
    The data stored in `record[project_key][label_key]`. This can be
    any json-serializable value.

    Raises
    ======
    ValueError if `on_missing="error"` and a record does not meet the
    number of `min_annotations` to be valid.
    """
    label = record[project_key][label_key]
    if min_annotations > 1:
        annotations = len(record[project_key][annotations_key])
        if annotations < min_annotations:
            if on_missing == "error":
                raise ValueError(
                    "Not enough annotations for this record. Minimum number "
                    f"of required annotations: {min_annotations} (got "
                    f"{annotations}). \n Full Record: \n {record}")
            else:
                return None
    return label
