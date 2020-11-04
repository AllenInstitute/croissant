import os.path
import json
from typing import List, Union
from pathlib import Path
import warnings
import time

import argschema
import marshmallow as mm
import pandas as pd
import lims.query_utils as qu
import h5py
from sklearn.model_selection import train_test_split

from croissant.ingest import annotation_df_from_file
from croissant.roi import LimsRoiSchema
from croissant.utils import (
    _munge_traces, json_write_local_or_s3, path_join_s3_os)


class SlappUploadManifest(mm.Schema):
    class Meta:
        unknown = mm.EXCLUDE

    experiment_id = mm.fields.Integer(
        required=True,
        description="Global experiment ID used by LIMS."
    )
    binarized_rois_path = argschema.fields.InputFile(
        required=True,
        description="Path to binarized ROIs output json on accessible FS."
    )
    traces_h5_path = mm.fields.Str(
        required=True,
        description="Path to raw trace file on accessible FS.",
        validate=os.path.exists
    )
    local_to_global_roi_id_map = mm.fields.Dict(
        keys=mm.fields.Str,
        values=mm.fields.Int,
        required=True
    )
    movie_path = mm.fields.Str(
        required=True,
        description="Path to raw movie file on accessible FS."
    )


class AnnotationIngestJobOutput(argschema.ArgSchema):
    train_data_path = mm.fields.Str(
        required=True,
        description="Local path or S3 URI to training data."
    )
    test_data_path = mm.fields.Str(
        required=True,
        description="Local path or S3 URI to test data."
    )
    created_at = mm.fields.Int(
        required=True,
        description="Seconds since epoch the files were created."
    )
    total_annotations = mm.fields.Int(
        required=True,
        description="Total number of annotations."
    )
    missing_data = mm.fields.List(
        mm.fields.Int,
        required=True,
        default=[],
        description="List of experiment_ids with missing upload manifests."
    )
    # TODO
    # label_summary = mm.fields.Dict(
    #     required=False,
    #     description="Metadata about annotations."
    # )


class AnnotationIngestJobInput(argschema.ArgSchema):
    """
    The results from loading in the manifests will be made available
    in the argschema parser args via the key "manifest_data".
    """
    slapp_upload_manifest_path = argschema.fields.InputFile(
        required=False,
        default=None,
        missing=None,
        description=("Path to the manifest file used to upload data "
                     "for SLAPP annotation tool. If missing will crawl "
                     "directories in data_root_dir looking for upload "
                     "manifests."))
    data_root_dir = argschema.fields.InputDir(
        required=False,
        missing=None,
        default=None,
        description=("Path to the root directory to search for individual "
                     "manifest files named `manifest_stub`. Used if "
                     "`slapp_upload_manifest_path` not provided.")
    )
    manifest_stub = argschema.fields.Str(
        required=False,
        default="slapp_tform_manifest.json",
        missing="slapp_tform_manifest.json",
        description=("Filename for the transform manifest per experiment. "
                     "Only used if `data_root_dir` is provided, and "
                     "slapp_upload_manifest_path isn't.")
    )
    output_location = argschema.fields.Str(
        required=True,
        description=("Path to a directory for saving the training data "
                     "on local file system, or an s3 URI for cloud blob "
                     "storage.")
    )
    test_split = argschema.fields.Float(
        required=False,
        missing=None,
        default=None,
        description=("Percent split for test data. Must be betweeen "
                     "0 and 1 (exclusive). Test data will be saved "
                     "in the `output_location` as train.json, and test "
                     "data will be saved as test.json."),
        validate=lambda x: ((x > 0.) and (x < 1.))
    )
    annotation_output_manifest = argschema.fields.Str(
        required=True,
        description=("S3 URI to the output directory containing annotations "
                     "from Sagemaker GT job."))
    labeling_project_key = argschema.fields.Str(
        required=True,
        description=("Name of this particular Sagemaker GT labeling job."))
    annotation_id_key = argschema.fields.Str(
        required=False,
        default="roi-id",
        missing="roi-id",
        description=("Key to use as index for annotation records in the "
                     "output manifest. Should be a top level key in record."))
    annotation_experiment_id_key = argschema.fields.Str(
        required=False,
        default="experiment-id",
        missing="experiment-id",
        description=("Key to use as index for annotation records in the "
                     "output manifest. Should be a top level key in record."))
    label_key = argschema.fields.Str(
        required=False,
        default="majorityLabel",
        missing="majorityLabel",
        description=("Key that contains the final labeled value to extract, "
                     "in the object (dict) indexed by `labeling_project_key` "
                     "in the output manifest."))
    # TODO
    # annotations_key = argschema.fields.Str(
    #     required=False,
    #     default=None,
    #     description=(
    #       "If applicable, the key in the record that contains the "
    #       "list of individual worker annotations (as dict records)."
    #       " Used to provide additional descriptive statistics "
    #       "about worker decisions."))
    label_map = argschema.fields.Dict(
        required=False,
        missing=None,
        default=None,
        description=("Map to apply to the labels. If no map provided, the "
                     "the raw values will be extracted from the manifest. "
                     "This can be skipped if the class labels are already "
                     "numeric (e.g. 0, 1...)"))

    @mm.post_load
    def make_objects(self, data, **kwargs):
        if data["slapp_upload_manifest_path"]:
            with open(data["slapp_upload_manifest_path"], "r") as f:
                upload_manifest_data = json.load(f)
            upload_manifest_obj = SlappUploadManifest().load(
                upload_manifest_data)
            data["manifest_data"] = [upload_manifest_obj]
        return data


class AnnotationIngestJob(argschema.ArgSchemaParser):
    default_schema = AnnotationIngestJobInput
    default_output_schema = AnnotationIngestJobOutput

    @staticmethod
    def compile_manifests(
            exp_ids: List[int], data_root_dir: Union[str, Path],
            manifest_stub: str):
        missing_data = []
        manifests = []
        if isinstance(data_root_dir, str):
            data_root_dir = Path(data_root_dir)
        for id_ in exp_ids:
            subdir = data_root_dir / f"{id_}"
            if os.path.exists(subdir / manifest_stub):
                with open(subdir / manifest_stub, "r") as f:
                    manifest_data = json.load(f)
                    manifest_obj = SlappUploadManifest().load(manifest_data)
                    manifests.append(manifest_obj)
            else:
                missing_data.append(id_)
        if missing_data:
            warnings.warn(f"Missing data for {len(missing_data)} ids: \n"
                          f"{missing_data}")
        return manifests, missing_data

    @staticmethod
    def get_metadata(exp_ids: List[int]):
        """This will only work when run on AIBS local network."""
        creds = qu.get_db_credentials()
        conn = qu.DbConnection(**creds)
        # Build array selector for IDs in query
        id_arr = ", ".join(map(str, exp_ids))
        results = conn.query(
            f"""
            SELECT
                movie_frame_rate_hz,
                e.name AS rig,
                d.full_genotype,
                st.acronym AS targeted_structure,
                i.depth,
                oe.id AS experiment_id
            FROM ophys_experiments oe
            JOIN ophys_sessions os ON oe.ophys_session_id = os.id
            JOIN equipment e ON os.equipment_id = e.id
            JOIN specimens sp ON os.specimen_id = sp.id
            JOIN donors d ON sp.donor_id = d.id
            JOIN imaging_depths i ON i.id=oe.imaging_depth_id
            JOIN structures st ON st.id=oe.targeted_structure_id
            WHERE oe.id IN ({id_arr})""")
        return results

    def main(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop("log_level"))
        additional_keys = [
            (self.args["annotation_id_key"], ),
            (self.args["annotation_experiment_id_key"], )]

        # Pull annotations from s3 or file system, and set global ID index
        annotations = annotation_df_from_file(
            self.args["annotation_output_manifest"],
            self.args["labeling_project_key"],
            self.args["label_key"],
            additional_keys=additional_keys)
        self.logger.info(
            "Annotations successfully ingested from output manifest "
            f"(n={len(annotations)}).")
        if self.args["label_map"]:
            annotations["label"] = annotations[self.args["label_key"]].map(
                self.args["label_map"])
        else:
            annotations["label"] = annotations[self.args["label_key"]].copy()
        # Format correctly
        annotations.set_index(self.args["annotation_id_key"], inplace=True)
        annotations.rename(
            {self.args["annotation_experiment_id_key"]: "experiment_id"},
            axis=1, inplace=True)
        exp_ids = list(set(annotations["experiment_id"]))

        # If an explicit manifest hasn't been passed, find all by looking
        # in subdirectories named for experiment_ids
        if not self.args.get("manifest_data"):
            self.args["manifest_data"], missing_data = self.compile_manifests(
                exp_ids,
                self.args["data_root_dir"],
                self.args["manifest_stub"])

        # Pull metadata from db
        # Contains rig, movie frame rate, targeted structure, genotype, depth
        self.logger.info("Retrieving metadata from LIMS database.")
        metadata = (pd.DataFrame(self.get_metadata(exp_ids))
                    .set_index("experiment_id"))

        # Load the data required
        # movie shape (for coo matrix rehydration)
        self.logger.info("Loading raw data files.")
        roi_raw_data = []
        for manifest in self.args["manifest_data"]:
            with h5py.File(manifest["movie_path"], "r") as f:
                image_shape = f["data"].shape[1:]
            with open(manifest["binarized_rois_path"], "r") as f:
                rois = json.load(f)
            local_ids = [i["id"] for i in rois]
            # roi coo matrix
            for local_id, global_id in (
                    manifest["local_to_global_roi_id_map"].items()):
                local_roi = LimsRoiSchema().load(
                    rois[local_ids.index(int(local_id))])
                # trace (downsampled)
                trace = _munge_traces(
                        [local_roi],
                        trace_file_path=manifest['traces_h5_path'],
                        trace_data_key='data',
                        trace_names_key='roi_names',
                        trace_sampling_rate=int(
                            metadata.loc[manifest["experiment_id"]]
                            ["movie_frame_rate_hz"]),
                        desired_trace_sampling_rate=4)[0]
                roi_raw_data.append({
                    "roi_id": global_id,
                    "coo_rows": local_roi["coo_roi"].row.tolist(),
                    "coo_cols": local_roi["coo_roi"].col.tolist(),
                    "coo_data": local_roi["coo_roi"].data.tolist(),
                    "image_shape": image_shape,
                    "trace": trace.tolist()})

        self.logger.info("Raw data loaded. Formatting data.")
        # Get things joined up and in the right format
        roi_dataframe = pd.DataFrame(roi_raw_data).set_index("roi_id")
        annotations = annotations.join(roi_dataframe, how="inner")
        annotations = annotations.join(
            metadata, on="experiment_id", how="inner")
        # For RoiWithMetadata
        annotations.drop(["movie_frame_rate_hz", "majorityLabel"],
                         axis=1, inplace=True)
        annotations["roi_id"] = annotations.index
        annotations.reset_index(drop=True, inplace=True)

        train_path = path_join_s3_os(
            self.args["output_location"], "train.json")
        test_path = path_join_s3_os(self.args["output_location"], "test.json")

        train, test = train_test_split(
            annotations, test_size=self.args["test_split"], shuffle=True)
        # Save s3 or local storage
        json_write_local_or_s3(train.to_dict(orient="records"), train_path)
        json_write_local_or_s3(test.to_dict(orient="records"), test_path)

        self.logger.info(f"Wrote data to {train_path} and {test_path}.")
        self.output({
            "train_data_path": train_path,
            "test_data_path": test_path,
            "created_at": int(time.time()),
            "total_annotations": len(annotations),
            "missing_data": missing_data
        })


if __name__ == "__main__":
    job = AnnotationIngestJob()
    job.main()
