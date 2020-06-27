import argschema
import pytest
from croissant.schemas import TrainingSchema
import json


@pytest.mark.parametrize("artifact_uri_is_s3", [True, False])
def test_artifact_uri(artifact_uri_is_s3, tmp_path):
    """tests that the post-load distinction between s3 and local is invoked
    """
    if artifact_uri_is_s3:
        artifact_uri = "s3://mybucket"
    else:
        artifact_uri = str(tmp_path)

    training_data_path = tmp_path / "my_train_data.json"
    with open(str(training_data_path), "w") as f:
        f.write("content")

    args = {
            "experiment_name": "my_experiment",
            "training_data": str(training_data_path),
            "artifact_uri": artifact_uri
            }
    mod = argschema.ArgSchemaParser(
            schema_type=TrainingSchema,
            input_data=args,
            args=[])
    for k, v in args.items():
        assert mod.args[k] == v


@pytest.mark.parametrize(
        "mode, expected",
        [
            ("default", TrainingSchema().dump_fields['param_grid'].default),
            ("from_file", {'some': 'stuff', '123': 'abc'}),
            ('from_dict', {'some': 'other', 'stuff': 123})])
def test_param_grid(mode, expected, tmp_path):
    training_data_path = tmp_path / "my_train_data.json"
    with open(str(training_data_path), "w") as f:
        f.write("content")

    args = {
            "experiment_name": "my_experiment",
            "training_data": str(training_data_path),
            "artifact_uri": "s3://somewhere"
            }

    if mode == 'from_file':
        pgfile = tmp_path / "pgrid.json"
        with open(pgfile, 'w') as f:
            json.dump(expected, f)
        args['param_grid_path'] = str(pgfile)

    if mode == "from_dict":
        args['param_grid'] = expected

    mod = argschema.ArgSchemaParser(
            schema_type=TrainingSchema,
            input_data=args,
            args=[])
    assert mod.args['param_grid'] == expected
