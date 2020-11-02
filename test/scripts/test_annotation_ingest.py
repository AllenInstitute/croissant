import json
from contextlib import nullcontext as does_not_raise

import argschema
import marshmallow as mm
import pytest

from croissant.scripts.annotation_ingest import (
    AnnotationIngestJobInput)


@pytest.fixture
def upload_manifest(request, tmp_path):
    # Set up all the temp files
    param = json.loads(request.param)
    binarized_rois_path = tmp_path / param["binarized_rois_path"]
    traces_h5_path = tmp_path / param["traces_h5_path"]
    manifest_path = tmp_path / "upload_manifest.json"
    manifest = param.copy()
    manifest["binarized_rois_path"] = str(binarized_rois_path)
    manifest["traces_h5_path"] = str(traces_h5_path)
    with open(binarized_rois_path, "w") as f:
        json.dump({}, f)
    with open(traces_h5_path, "w") as f:
        json.dump({}, f)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    yield str(manifest_path)


@pytest.fixture
def metadata_query():
    return [
        {"movie_frame_rate_hz": 11.0,
         "rig": "MESO.1",
         "full_genotype": "Sst-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
         "id": 1060223946},
        {"movie_frame_rate_hz": 11.0,
         "rig": "MESO.1",
         "full_genotype": "Sst-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
         "id": 1060223947}]


@pytest.mark.parametrize(
    "upload_manifest, context", [
        (
            """{
                "experiment_id": 123,
                "binarized_rois_path": "binarize.json",
                "traces_h5_path": "trace.h5",
                "local_to_global_roi_id_map": {
                    "123": 444,
                    "465": 999
                },
                "movie_path": "movie.h5",
                "extra_field": "https://www.youtube.com/watch?v=oHg5SJYRHA0"
            }""",
            does_not_raise()
        ),
        (
            """{
                "experiment_id": 999,
                "binarized_rois_path": "binarize.json",
                "traces_h5_path": "trace.h5"
            }""",
            pytest.raises(mm.exceptions.ValidationError)
        )], indirect=["upload_manifest"],
    ids=["well_formed", "malformed"]
)
def test_annotation_job_schema_make_objects(
        upload_manifest, context, tmp_path):
    """Test that the objects are properly loaded and turned into
    appropriate objects."""
    input_dict = {"slapp_upload_manifest_path": upload_manifest,
                  "annotation_output_location": str(tmp_path),
                  "labeling_project_key": "astley",
                  "annotation_id_key": "id",
                  "output_location": str(tmp_path)}
    with context:
        result = argschema.ArgSchemaParser(
            schema_type=AnnotationIngestJobInput,
            input_data=input_dict, args=[])
        assert "manifest_data" in result.args


