from typing import Optional, Any, Union, List, Tuple
from pathlib import Path
import pandas as pd
from functools import partial
import argparse

from croissant.utils import nested_get_item, read_jsonlines


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
    reader = read_jsonlines(filepath)
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


def _ingest_uri(
        filepath: Union[str, Path],
        project_key: str,
        label_key: str,
        roi_id_key: Optional[Tuple[str]] = None,
        annotations_key: Optional[Tuple[str]] = None,
        min_annotations: int = 1,
        on_missing: str = "skip",
        drop_na: bool = False) -> pd.DataFrame:
    """Main entry for processing the data. See
    `croissant.ingest.annotation_df_from_file` for most function
    parameters.

    Parameters
    ==========
    filepath: str
    project_key: str
    label_key: str
    annotations_key: Optional[Tuple[str]]
        Key path to annotations key. Should be in the top level of the
        project_key dictionary.
    roi_id_key: Optional[str]
        Top-level key for ROI id in the json record
    min_annotations: str
    on_missing: str
    drop_na: bool (default=False)
        Whether to drop labels that have null values in the output.

    Returns
    =======
    pd.DataFrame of the following format:
        label: int
            Column of labels. 0 if "not cell", 1 if "cell".
        roi_id: int (if `roi_id_key` is passed)
            The ROI's id.
        annotations: List[int] (if `annotations_key` is passed)
            List of annotations from workers. 0 if "not cell", 1 if
            "cell".
    """
    return_cols = ["label"]
    additional_keys = []
    if roi_id_key:
        roi_out_key = roi_id_key[-1]        # Need the lowest level key if exists for output   # noqa
        additional_keys.append(roi_id_key)
        return_cols.append("roi_id")
    annot_out_key = None
    if annotations_key:
        annot_out_key = annotations_key[-1]     # Need the lowest level key if exists for parse_annotations_df      # noqa
        additional_keys.append(annotations_key)
        return_cols.append("annotations")

    output = annotation_df_from_file(
        filepath, project_key, label_key, annot_out_key,
        min_annotations, on_missing, additional_keys)
    output["label"] = output[label_key].map({"cell": 1, "not cell": 0})
    if annot_out_key:         # Map the annotations same way as labels
        output["annotations"] = output[annotations_key[-1]].apply(
            lambda annotations: [1 if annot["roiLabel"] == "cell" else 0
                                 for annot in annotations])
    if roi_id_key:
        output.rename(columns={roi_out_key: "roi_id"}, inplace=True)
    if drop_na:
        output.dropna(subset=["label"], inplace=True)
    return output[return_cols]


def _keypath(s):
    """Parser for keypath from command line args."""
    return tuple(s.split(","))


def _make_parser():
    """ Make the command line argument parser when module is run as main.
    Split out for easier testing.
    """
    parser = argparse.ArgumentParser("annotationIngest")
    parser.add_argument(
        "manifest_uri", type=str,
        help=("S3 URI or local filepath to the output manifest file of a "
              "SageMaker GT job, in jsonlines format."))
    parser.add_argument(
        "output_csv", type=str,
        help="s3 URI or local filepath to save the results (csv format).")
    parser.add_argument(
        "project_key", type=str,
        help="Top-level key for the output manifest data")
    parser.add_argument(
        "label_key", type=str,
        help=("Key for the label data in each output manifest "
              " record[project_key]."))
    parser.add_argument(
        "--min_annotations", type=int, default=1,
        help="Optional required number of annotations to enforce per record.")
    parser.add_argument(
        "--on_missing", type=str, default="skip", choices=["skip", "error"],
        help=("What do to if required number of annotations is not met. "
              "if 'skip', replace the label with null. If error, will raise "
              "an error instead."))
    parser.add_argument(
        "--roi_id_key", type=_keypath, default=None,
        help=("comma-separated 'key path' to ROI id in the manifest records. "
              "Example: `--roi_id_key project,roi-id` will access "
              "record['project']['roi-id']"))
    parser.add_argument(
        "--annotations_key", type=_keypath, default=None,
        help=("Comma-separated 'key path' to worker annotations list in the "
              "manifest records. The worker annotations must be a list where "
              "each element is a dictionary record with the following format "
              ": {'workerId': <string>, 'roiLabel': <string> }"))
    parser.add_argument(
        "--drop_na", action="store_true", default=False,
        help=("Flag to drop records with null labels in the output data."))
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    output = _ingest_uri(
        args.manifest_uri, args.project_key, args.label_key,
        roi_id_key=args.roi_id_key, annotations_key=args.annotations_key,
        min_annotations=args.min_annotations, on_missing=args.on_missing,
        drop_na=args.drop_na)
    output.to_csv(args.output_csv)    # will use s3fs if s3 uri is passed
