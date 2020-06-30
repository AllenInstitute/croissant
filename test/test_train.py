from pathlib import Path
import tempfile
import json
import os

import pytest
from unittest.mock import MagicMock
import mlflow
import mlflow.sklearn
import joblib
from functools import partial
import subprocess

import croissant.train as train
from croissant.features import FeatureExtractor


@pytest.fixture()
def train_data():
    yield Path(__file__).parent / 'resources' / 'dev_train_rois.json'


@pytest.fixture()
def test_data():
    yield Path(__file__).parent / 'resources' / 'dev_test_rois.json'


@pytest.fixture
def mock_classifier(request):
    mock_clf = MagicMock()
    mock_clf.best_score_ = request.param['best_score']
    mock_clf.best_params_ = request.param['best_params']
    mock_clf.param_grid = request.param['param_grid']
    mock_clf.cv_results_ = request.param['results']
    mock_clf.scorer_ = {request.param['scorer']: 'sklearn_callable'}
    mock_clf.best_estimator_ = request.param['pickled']
    return mock_clf


@pytest.mark.parametrize(
        "search_grid", [{'model__l1_ratio': [0.25, 0.5, 0.75]}])
def test_train_classifier(search_grid, train_data, test_data, tmp_path):
    """tests that `train_classifier()` generates a classifier that
    makes binary predictions.
    """
    clf = train.train_classifier(training_data_path=train_data,
                                 param_grid=search_grid,
                                 scoring=['roc_auc'],
                                 refit='roc_auc')

    # load testing roi and generate input features
    with open(test_data, 'r') as open_testing:
        testing_data = json.load(open_testing)
    features = FeatureExtractor.from_list_of_dict(testing_data).run()

    predictions = clf.predict(features)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize(
        "mock_classifier",
        [
            ({
                'best_score': 123,
                'best_params': {'a': 1, 'b': 2},
                'param_grid': {'p1': [4, 5, 6.7]},
                'scorer': "metric1",
                'results': {
                    'mean_test_metric1': [0.1, 0.2, 0.2],
                    'std_test_metric1': [0.1, 0.2, 0.2]},
                'pickled': {'something': ['to', 'pickle']}
                }),
            ], indirect=["mock_classifier"])
def test_mlflow_log_classifier(tmp_path, mock_classifier):
    """with a mocked classifier, tests that `mlflow_log_classifier()` logs
    to mlflow
    """
    with tempfile.TemporaryDirectory() as temp_out_dir:
        experiment_name = "myexperiment"
        tracking_uri = str(Path(temp_out_dir).as_uri())
        artifact_uri = str(Path(tmp_path) / "artifacts")
        training_data_path = "some_string"

        run_id = train.mlflow_log_classifier(
                experiment_name,
                tracking_uri,
                artifact_uri,
                training_data_path,
                mock_classifier)

        # check that this call adds a run to mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        run_infos = client.list_run_infos(experiment.experiment_id)
        run_ids = [r.run_id for r in run_infos]
        assert run_id in run_ids

        # check that this run has the right stuff
        myrun = client.get_run(run_id)
        # metrics
        assert myrun.data.metrics['Best_Score'] == mock_classifier.best_score_
        scorer = list(mock_classifier.scorer_.keys())[0]
        for s in ['Mean', 'STD']:
            assert f"{s}_{scorer}" in myrun.data.metrics
        # tags
        assert myrun.data.tags['training_data_path'] == training_data_path
        assert myrun.data.tags['param_grid'] == \
            repr(mock_classifier.param_grid)
        # parameters
        for k, v in mock_classifier.best_params_.items():
            assert myrun.data.params[k] == repr(v)
        # artifact
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        model = "trained_model.joblib"
        assert model in artifact_paths
        my_artifact_path = Path(myrun.info.artifact_uri) / model
        assert my_artifact_path.exists()
        unpickled = joblib.load(my_artifact_path)
        assert unpickled == mock_classifier.best_estimator_


def test_ClassifierTrainer(train_data, tmp_path, monkeypatch):
    """tests argschema entry point with mocked training and logging
    """
    args = {
            "training_data": str(train_data),
            "param_grid": {'a': [1, 2, 3]},
            'scoring': ['a'],
            'refit': 'a',
            'experiment_name': 'my_experiment',
            'mlflow_tracking_uri': str(tmp_path),
            'artifact_uri': str(tmp_path),
            }

    mock_classifier = MagicMock()
    mock_classifier.best_score_ = "a good score"
    mock_classifier.best_params_ = "some good params"
    mock_train_classifier = MagicMock(return_value=mock_classifier)
    mock_mlflow_log_classifier = MagicMock()
    mpatcher = partial(monkeypatch.setattr, target=train)
    mpatcher(name="train_classifier", value=mock_train_classifier)
    mpatcher(name="mlflow_log_classifier", value=mock_mlflow_log_classifier)

    ctrain = train.ClassifierTrainer(input_data=args, args=[])
    ctrain.train()

    mock_train_classifier.assert_called_once_with(
            training_data_path=train_data,
            param_grid=args['param_grid'],
            scoring=args['scoring'],
            refit=args['refit'])

    mock_mlflow_log_classifier.assert_called_once_with(
            args['experiment_name'],
            args['mlflow_tracking_uri'],
            args['artifact_uri'],
            args['training_data'],
            mock_classifier)
