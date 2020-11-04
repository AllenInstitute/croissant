from pathlib import Path
import json
import os
from functools import partial
from urllib.parse import urlparse
from unittest.mock import MagicMock

import pytest
import mlflow
from moto import mock_s3
import boto3
import argschema
import marshmallow as mm
import sklearn

from hyperopt import hp
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

import croissant.train as train
from croissant.features import FeatureExtractor


@pytest.fixture()
def X_y():
    """Random dataset and features"""
    return load_breast_cancer(return_X_y=True)


@pytest.fixture()
def train_data():
    tp = Path(__file__).parent / 'resources' / 'dev_train_rois.json'
    yield str(tp)


@pytest.fixture()
def test_data():
    tp = Path(__file__).parent / 'resources' / 'dev_test_rois.json'
    yield str(tp)


@pytest.fixture
def mock_classifier(request):
    mock_clf = MagicMock()
    mock_clf.n_test = request.param['n_test_values']
    mock_clf.pickled = request.param["pickled"]
    mock_clf.predict = MagicMock(return_value=[0] * mock_clf.n_test)
    return mock_clf


@pytest.fixture()
def mock_FeatureExtractor():
    mock_FE = MagicMock()
    mock_FE.run = MagicMock()
    mock_FE.from_list_of_dict = MagicMock(return_value=mock_FE)
    yield mock_FE


@pytest.fixture
def experiment_name():
    return "myexperiment"


@pytest.fixture
def mlflow_client(experiment_name, tmp_path):
    """
    MLFlow can't nicely be mocked to stay in-memory, so we need to
    do some setup for testing.

    In production, we plan to run from the command line with
    `mlflow run ...` This setup block is equivalent to the following
    mlflow CLI call:

    > mlflow experiments create \
            --experiment-name <ename> \
            --artifact-location <path>

    and then, when running from the MLProject file
    > mlflow run . --experiment-name <ename> ...

    mlflow run reads the tracking URI from the env variable
    MLFLOW_TRACKING_URI
    """
    experiment_name = experiment_name
    artifact_uri = str(tmp_path / "artifacts")
    tracking_uri = str(tmp_path / "tracking")
    os.environ['MLFLOW_TRACKING_URI'] = str(tracking_uri)
    mlflow.create_experiment(
            experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient(
        tracking_uri=str(tracking_uri))
    yield client


@pytest.mark.parametrize(
    "drop_cols", [None, ["full_genotype"]]
)
@pytest.mark.parametrize(
    "model", ["RandomForestClassifier", "LogisticRegression"]
)
def test_train_classifier_binary_preds(
        model, train_data, test_data, drop_cols, tmp_path):
    """tests that `train_classifier()` generates a classifier that
    makes binary predictions.
    """
    clf, _, _, _ = train.train_classifier(
        model=model,
        training_data_path=train_data,
        test_data_path=test_data,
        scorer='roc_auc',
        max_iter=2,
        optimizer="rand",
        n_folds=2,
        refit=True,
        drop_cols=drop_cols)

    # load testing roi and generate input features
    with open(test_data, 'r') as open_testing:
        testing_data = json.load(open_testing)
    features = FeatureExtractor.from_list_of_dict(testing_data).run()

    predictions = clf.predict(features)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize(
    "model", ["RandomForestClassifier", "LogisticRegression"]
)
@pytest.mark.parametrize(
    "scorer,n_folds,test_metrics,optimizer,refit",
    [
        ("accuracy", 3, ["roc_auc", "average_precision"], "rand", True),
        ("recall", 2, None, "tpe", False),
    ]
)
def test_train_classifier_integration(
        train_data, test_data, model, scorer, n_folds, test_metrics,
        optimizer, refit):
    """Are we playing nicely with sklearn, etc."""
    actual_pipe, actual_opt_score, actual_metrics, actual_cmat = (
        train.train_classifier(
            model,
            train_data,
            test_data,
            scorer,
            max_iter=1,
            optimizer=optimizer,
            test_metrics=test_metrics,
            n_folds=n_folds,
            seed=42,
            refit=refit
        )
    )

    # test correct number of values for optimization metric folds
    assert n_folds == len(actual_opt_score[1])
    # test correct scorer
    assert scorer == actual_opt_score[0]
    # test correct test metrics evaluated
    if test_metrics is not None:
        expected_test_metrics = [f"test_{m}" for m in test_metrics]
        assert len(actual_metrics) == len(expected_test_metrics)
        assert set(actual_metrics).issubset(expected_test_metrics)
    else:
        assert list(actual_metrics) == [f"test_{scorer}"]
    # test chose the correct model
    assert model in str(actual_pipe[-1].__class__)
    # test confusion matrix formatted correctly
    assert len(actual_cmat) == 4
    assert set(actual_cmat).issubset(
        ["test_TN", "test_TP", "test_FN", "test_FP"])


@pytest.mark.parametrize(
    "y_true, y_pred, prefix, expected",
    [
        (
            [1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1],
            "",
            {"TN": 1, "TP": 3, "FN": 0, "FP": 1}
        ),
        (
            [0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            "test_",
            {"test_TN": 0, "test_TP": 2, "test_FN": 0, "test_FP": 4}
        )
    ]
)
def test_binary_confusion_dict(y_true, y_pred, prefix, expected):
    actual = train._binary_confusion_dict(y_true, y_pred, prefix=prefix)
    assert expected == actual


@pytest.mark.parametrize(
    "y_true,y_pred,",
    [([0, 1, 1, 1, 2], [0, 0, 0, 1, 1])]
)
def test_binary_confusion_dict_raises_error_non_binary(y_true, y_pred):
    with pytest.raises(ValueError, match=r"Must have binary labels"):
        train._binary_confusion_dict(y_true, y_pred)


@pytest.mark.parametrize(
    "tags,metrics,confusion_matrix",
    [
        (
            {"never": "gonna", "give": "you"},
            {"up": 0.867, "never": 0.5309},
            {"TP": 13, "TN": 15, "FP": 1, "FN": 0},
        ),
        (None, None, None)      # Testing optional args
    ]
)
@pytest.mark.parametrize(
        "train_path,test_path,mock_classifier,cv_scores,expected_cv",
        [
            (
                "/fake/path/file.txt",
                "s3://fake/path/file_2.txt",
                {
                    "n_test_values": 100,
                    "pickled": {"a": "b"}
                },
                ("accuracy", [0.9, 0.8, 0.7]),
                {"split0_train_accuracy": 0.9,
                    "split1_train_accuracy": 0.8,
                    "split2_train_accuracy": 0.7}
            )
        ], indirect=["mock_classifier"])
def test_mlflow_log_classifier(
        tmp_path, train_path, test_path, mock_classifier, cv_scores,
        metrics, confusion_matrix, tags, expected_cv, monkeypatch,
        experiment_name, mlflow_client):
    """
    Test mlflow logging behavior with mocked classifier
    """
    def _mock_dump(obj, path):
        with open(path, "w") as f:
            json.dump(obj.pickled, f)

    monkeypatch.setattr(train.joblib, "dump", _mock_dump)
    run_id = train.mlflow_log_classifier(
            train_path,
            test_path,
            mock_classifier,
            cv_scores,
            metrics,
            confusion_matrix,
            tags)

    # check that this call adds a run to mlflow
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    run_infos = mlflow_client.list_run_infos(experiment.experiment_id)
    run_ids = [r.run_id for r in run_infos]
    assert run_id in run_ids

    # check that this run has the right stuff
    myrun = mlflow_client.get_run(run_id)

    # metrics
    combo_expected_metrics = {}
    combo_expected_metrics.update(metrics or {})
    combo_expected_metrics.update(confusion_matrix or {})
    combo_expected_metrics.update(expected_cv)
    for k, v in combo_expected_metrics.items():
        assert k in myrun.data.metrics
        assert myrun.data.metrics[k] == v

    # tags
    if tags is None:
        tags = {}       # make an iterable
    expected_tags = {
            "training_data_path": train_path,
            "test_data_path": test_path,
            "sklearn_version": sklearn.__version__,
            **tags
            }
    for k, v in expected_tags.items():
        assert k in myrun.data.tags
        assert myrun.data.tags[k] == v

    # artifact
    artifacts = mlflow_client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    model = "trained_model.joblib"
    assert model in artifact_paths
    my_artifact_path = Path(myrun.info.artifact_uri) / model
    assert my_artifact_path.exists()


def test_ClassifierTrainer(train_data, test_data, tmp_path, monkeypatch):
    """tests argschema entry point with mocked training and logging
    """
    args = {
        "training_data": str(train_data),
        "test_data": str(test_data),
        "scorer": "accuracy",
        "reported_metrics": ["roc_auc"],
        "model": "LogisticRegression",
        "max_iter": 2,
        "refit": True,
        "seed": 99,
        "optimizer": "rand",
        "n_folds": 3,
        "drop_cols": ["full_genotype"]
    }

    mock_classifier = MagicMock()
    mock_train_classifier = MagicMock(
        return_value=(
            mock_classifier,
            ("score", [0.9, 0.8, 0.7]),
            {"test_score": 0.8},
            {"TN": 1, "TP": 1, "FN": 1, "FP": 1}))

    tags = {"model": args["model"],
            "metric": args["scorer"],
            "seed": args["seed"],
            "n_folds": args["n_folds"],
            "max_iter": args["max_iter"],
            "optimizer": args["optimizer"],
            "refit": args["refit"]}

    mock_mlflow_log_classifier = MagicMock()
    mpatcher = partial(monkeypatch.setattr, target=train)
    mpatcher(name="train_classifier", value=mock_train_classifier)
    mpatcher(name="mlflow_log_classifier", value=mock_mlflow_log_classifier)

    ctrain = train.ClassifierTrainer(input_data=args, args=[])
    ctrain.train()

    mock_train_classifier.assert_called_once_with(
            model=args["model"],
            training_data_path=train_data,
            test_data_path=test_data,
            scorer=args["scorer"],
            max_iter=args["max_iter"],
            optimizer=args["optimizer"],
            test_metrics=args["reported_metrics"],
            n_folds=args["n_folds"],
            seed=args["seed"],
            refit=args["refit"],
            drop_cols=args["drop_cols"]
    )

    mock_mlflow_log_classifier.assert_called_once_with(
            args['training_data'],
            args['test_data'],
            mock_classifier,
            ("score", [0.9, 0.8, 0.7]),
            metrics={"test_score": 0.8},
            confusion_matrix={"TN": 1, "TP": 1, "FN": 1, "FP": 1},
            tags=tags)


@pytest.fixture(scope="module")
def s3_file():
    file_uri = "s3://myschematest/my/file.json"
    mock = mock_s3()
    mock.start()
    test_up = urlparse(file_uri)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=test_up.netloc)
    resp = s3.put_object(Bucket=test_up.netloc, Key=test_up.path[1:],
                         Body=json.dumps({'a': 1}).encode('utf-8'))
    print(resp)
    yield file_uri
    mock.stop()


class TestTrainingSchema:

    def test_training_schema_works(self, s3_file):
        args = {
            "training_data": s3_file,
            "test_data": s3_file
        }
        # uris exist, no errors
        argschema.ArgSchemaParser(input_data=args,
                                  schema_type=train.TrainingSchema,
                                  args=[])

    def test_training_schema_fails_nonexistent_data(self, s3_file):
        args = {
            "training_data": "s3://myschematest/does/not/exist.txt",
            "test_data": s3_file
        }
        with pytest.raises(mm.ValidationError, match=r".*does not exist"):
            argschema.ArgSchemaParser(input_data=args,
                                      schema_type=train.TrainingSchema,
                                      args=[])

    def test_training_schema_fails_bad_model(self, s3_file):
        args = {
            "training_data": s3_file,
            "test_data": s3_file,
            "model": "SomeRandomAlgorithm"
        }
        with pytest.raises(mm.ValidationError, match=r"Must be one of"):
            argschema.ArgSchemaParser(input_data=args,
                                      schema_type=train.TrainingSchema,
                                      args=[])

    def test_training_schema_fails_bad_optimizer(self, s3_file):
        args = {
            "training_data": s3_file,
            "test_data": s3_file,
            "optimizer": "CoolOptimizer"
        }
        with pytest.raises(mm.ValidationError, match=r"Must be one of"):
            argschema.ArgSchemaParser(input_data=args,
                                      schema_type=train.TrainingSchema,
                                      args=[])


@pytest.mark.parametrize(
    "param_space, expected_params", [
        (None, ["bootstrap", "max_depth", "max_features", "min_samples_leaf",
                "min_samples_split", "n_estimators", "oob_score"]),
        ({"max_depth":
          hp.choice("max_depth", [1, 5, 10])}, ["max_depth"])
    ]
)
@pytest.mark.parametrize(
    "pipeline,n_splits,scorer,low_good",
    [
        (None, 5, "roc_auc", False,),
        (Pipeline(steps=[("scaler", StandardScaler())]), 3,
         make_scorer(brier_score_loss), True,),
    ]
)
def test_RandomForestTuner_fmin_fmax(
        X_y, param_space, pipeline, n_splits, scorer, low_good,
        expected_params):
    X, y = X_y
    tuner = train.RandomForestTuner(X, y, pipeline, param_space=param_space,
                                    n_splits=n_splits, scorer=scorer)
    if low_good:
        best_params = tuner.fmin(2)
    else:
        best_params = tuner.fmax(2)

    assert 2 == len(tuner.trials), "Did not evaluate correct number of trials."
    for param in expected_params:
        assert param in best_params.keys(), (
            f"Missing expected param {param} in best param set.")


@pytest.mark.parametrize(
    "param_space, expected_params", [
        (None, ["C", "penalty", "l1_ratio"]),
        ({"penalty": "l2",
          "solver": hp.choice("solver", ["newton-cg", "sag", "lbfgs"])},
         ["penalty", "solver"])
    ]
)
@pytest.mark.parametrize(
    "pipeline,n_splits,scorer,low_good",
    [
        (None, 5, "roc_auc", False,),
        (Pipeline(steps=[("scaler", StandardScaler())]), 3,
         make_scorer(brier_score_loss), True,),
    ]
)
def test_LogisticRegressionTuner_fmin_fmax(
        X_y, param_space, pipeline, n_splits, scorer, low_good,
        expected_params):
    X, y = X_y
    tuner = train.LogisticRegressionTuner(
        X, y, pipeline, param_space=param_space,
        n_splits=n_splits, scorer=scorer)
    if low_good:
        best_params = tuner.fmin(2)
    else:
        best_params = tuner.fmax(2)

    assert 2 == len(tuner.trials), "Did not evaluate correct number of trials."
    for param in expected_params:
        assert param in best_params.keys(), (
            f"Missing expected param {param} in best param set.")
