import argschema
import pytest
from croissant.schemas import TrainingSchema
import json


@pytest.mark.parametrize(
        "mode, expected",
        [
            ("default", TrainingSchema().fields['param_grid'].default),
            ("from_file", {'some': 'stuff', '123': 'abc'}),
            ('from_dict', {'some': 'other', 'stuff': 123})])
def test_param_grid(mode, expected, tmp_path):
    training_data_path = tmp_path / "my_train_data.json"
    with open(str(training_data_path), "w") as f:
        f.write("content")

    args = {
            "training_data": str(training_data_path),
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
