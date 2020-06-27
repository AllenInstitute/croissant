from pathlib import Path
import tempfile
import json

import pytest
import mlflow
import mlflow.sklearn
import argschema
import joblib

from croissant.train import train_classifier, mlflow_log_classifier
from croissant.roi import RoiWithMetadata
from croissant.features import FeatureExtractor


@pytest.fixture()
def train_data():
    yield Path(__file__).parent / 'resources' / 'dev_train_rois.json'


@pytest.fixture()
def test_data():
    yield Path(__file__).parent / 'resources' / 'dev_test_rois.json'


@pytest.mark.parametrize(
        "experiment_name, search_grid",
        [
            ('wst_exp', {'model__l1_ratio': [0.25, 0.5, 0.75]})
            ]
                        )
def test_train_classifier(experiment_name, search_grid, train_data, test_data,
                          tmp_path):
    with tempfile.TemporaryDirectory() as temp_out_dir:
        temp_uri = Path(temp_out_dir).as_uri()
        artifacts_dir = Path(tmp_path) / "artifacts"

        mlflow.set_tracking_uri(temp_uri)
        mlflow.create_experiment(
                experiment_name,
                str(artifacts_dir))
        mlflow.set_experiment(experiment_name)

        clf = train_classifier(training_data_path=train_data,
                                  param_grid=search_grid,
                                  scoring=['roc_auc'],
                                  refit='roc_auc')

        run_id = mlflow_log_classifier(train_data, clf)
        mlflow_run = mlflow.get_run(run_id)
        run_dir = Path(mlflow_run.info.artifact_uri)

        # verify the model exists
        model_path = run_dir / 'trained_model.joblib'
        assert model_path.exists()

        # validate MlFlow model can provide classification on training data
        sk_model = joblib.load(model_path)

        # load testing roi and generate input features
        with open(test_data, 'r') as open_testing:
            testing_data = json.load(open_testing)
        features = FeatureExtractor.from_list_of_dict(testing_data).run()

        predictions = sk_model.predict(features)
        assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize(
        "search_grid", [{'model__l1_ratio': [0.25, 0.5, 0.75]}])
def test_train_classifier2(search_grid, train_data, test_data, tmp_path):
    clf = train_classifier(training_data_path=train_data,
                              param_grid=search_grid,
                              scoring=['roc_auc'],
                              refit='roc_auc')

    # load testing roi and generate input features
    with open(test_data, 'r') as open_testing:
        testing_data = json.load(open_testing)
    features = FeatureExtractor.from_list_of_dict(testing_data).run()

    predictions = clf.predict(features)
    assert set(predictions).issubset({0, 1})
