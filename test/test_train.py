from pathlib import Path
import os
import tempfile
from urllib.parse import urlsplit
import json

import pytest
import mlflow
import mlflow.sklearn

from croissant.train import train_classifier
from croissant.features import FeatureExtractor, Roi, RoiMetadata


@pytest.fixture()
def data_paths():
    resource_path = Path(__file__).parent / 'resources'
    training_data_path = resource_path / 'dev_train_rois.json'
    testing_data_path = resource_path / 'dev_test_rois.json'
    yield training_data_path, testing_data_path


@pytest.mark.parametrize("environment, experiment_name, search_grid",
                         [('test', 'test_exp', {'model__l1_ratio': [0.25,
                                                                    0.5,
                                                                    0.75]})]
                         )
def test_train_classifier(environment, experiment_name,
                          search_grid, data_paths):
    with tempfile.TemporaryDirectory() as temp_out_dir:
        temp_uri = Path(temp_out_dir).as_uri()

        mlflow.set_tracking_uri(temp_uri)
        mlflow.set_experiment(experiment_name)

        train_classifier(environment=environment,
                         training_data=data_paths[0],
                         output_dir=Path(temp_out_dir),
                         search_grid=search_grid)

        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        artifact_path = urlsplit(mlflow_experiment.artifact_location).path
        if os.name == 'nt':
            artifact_path = artifact_path[1:]
        artifact_path = Path(artifact_path)
        run_dir = [artifact_path / path for path in os.listdir(artifact_path)
                   if os.path.isdir(artifact_path / path)][0]

        artifacts_dir = run_dir / 'artifacts'

        # verify the model exists
        model_dir = artifacts_dir / 'FittedModel'
        # inside Fitted Model there should be model.pkl, and MLmodel, and
        # a conda.yaml file
        expected_files = [model_dir / 'model.pkl', model_dir / 'MLmodel',
                          model_dir / 'conda.yaml']
        for expected_file in expected_files:
            assert expected_file.exists()

        # validate MlFlow model can provide classification on training data
        sk_model = mlflow.sklearn.load_model(model_dir.as_uri())

        # load testing roi and generate input features
        testing_data_path = data_paths[1]
        with open(testing_data_path, 'r') as open_testing:
            testing_data = json.load(open_testing)
            test_roi = testing_data[0]

        roi = [Roi(roi_id=test_roi['roi_id'],
                   coo_cols=test_roi['coo_cols'],
                   coo_rows=test_roi['coo_rows'],
                   coo_data=test_roi['coo_data'],
                   image_shape=test_roi['image_shape'])]
        roi_meta = [RoiMetadata(depth=test_roi['depth'],
                                full_genotype=test_roi['full_genotype'],
                                targeted_structure=test_roi['targeted_'
                                                            'structure'],
                                rig=test_roi['rig'])]
        trace = [test_roi['trace']]
        features = FeatureExtractor(rois=roi, metadata=roi_meta,
                                    dff_traces=trace).run()

        predictions = sk_model.predict(features)
        # binary classification assert it falls within that paradigm
        assert predictions[0] in [0, 1]
