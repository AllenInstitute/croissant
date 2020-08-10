from unittest.mock import MagicMock
import croissant.online_training as otr
import json
import pytest
import os


@pytest.fixture()
def temp_file(tmp_path):
    fp = tmp_path / "hello.txt"
    fp.write_text("hello world")
    yield str(fp)


def test_online_training_schema(tmp_path, temp_file):
    tmp_env_var = None
    if "MLFLOW_EXPERIMENT" in os.environ.keys():
        tmp_env_var = os.environ.pop("MLFLOW_EXPERIMENT")

    os.environ["MLFLOW_EXPERIMENT"] = "abcdef98765"

    outj = tmp_path / "output.json"
    training_args = {
            "training_data": temp_file,
            "test_data": temp_file,
            }
    args = {
            "output_json": str(outj),
            "training_args": training_args,
            "cluster": "mycluster",
            "taskDefinition": "mycluster",
            "container": "mycontainer",
            "subnet": "mysubnet",
            "securityGroup": "mysecurity",
            "trackingURI": "mytracking",
            }
    ot = otr.OnlineTraining(input_data=args, args=[])

    assert ot.args["experiment_name"] == os.environ["MLFLOW_EXPERIMENT"]

    if tmp_env_var is not None:
        os.environ["MLFLOW_EXPERIMENT"] = tmp_env_var


def test_online_training_schema_exception(tmp_path, temp_file):
    tmp_env_var = None
    if "MLFLOW_EXPERIMENT" in os.environ.keys():
        tmp_env_var = os.environ.pop("MLFLOW_EXPERIMENT")

    outj = tmp_path / "output.json"
    training_args = {
            "training_data": temp_file,
            "test_data": temp_file,
            }
    args = {
            "output_json": str(outj),
            "training_args": training_args,
            "cluster": "mycluster",
            "taskDefinition": "mycluster",
            "container": "mycontainer",
            "subnet": "mysubnet",
            "securityGroup": "mysecurity",
            "trackingURI": "mytracking",
            }
    with pytest.raises(otr.OnlineTrainingException):
        otr.OnlineTraining(input_data=args, args=[])

    if tmp_env_var is not None:
        os.environ["MLFLOW_EXPERIMENT"] = tmp_env_var


def test_online_training(tmp_path, temp_file, monkeypatch):
    outj = tmp_path / "output.json"
    training_args = {
            "training_data": temp_file,
            "test_data": temp_file,
            }
    args = {
            "output_json": str(outj),
            "training_args": training_args,
            "cluster": "mycluster",
            "taskDefinition": "mycluster",
            "container": "mycontainer",
            "subnet": "mysubnet",
            "securityGroup": "mysecurity",
            "trackingURI": "mytracking",
            "experiment_name": "myexperiment"
            }

    mock_boto = MagicMock()
    mock_client = MagicMock()
    mock_boto.client = MagicMock(return_value=mock_client)
    mock_client.run_task = MagicMock(return_value={"a": "1", "b": "2"})
    monkeypatch.setattr(target=otr, name="boto3", value=mock_boto)
    ot = otr.OnlineTraining(input_data=args, args=[])
    ot.run()

    with open(outj, 'r') as f:
        output = json.load(f)

    assert output == {'response': mock_client.run_task.return_value}
