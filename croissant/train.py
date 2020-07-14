from pathlib import Path
import logging
import marshmallow as mm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import argschema
import joblib
import tempfile
from typing import List
from urllib.parse import urlparse

from croissant.features import FeatureExtractor, feature_pipeline
from croissant.utils import json_load_local_or_s3, object_exists


logger = logging.getLogger('TrainClassifier')


class TrainingSchema(argschema.ArgSchema):
    training_data = argschema.fields.Str(
        required=True,
        description=("s3 uri or local path, <stem>.json containing a list "
                     "of dicts, where each dict can be passed into "
                     "RoiWithMetaData.from_dict()."))
    test_data = argschema.fields.Str(
        required=True,
        description=("s3 uri or local path, <stem>.json containing a list "
                     "of dicts, where each dict can be passed into "
                     "RoiWithMetaData.from_dict()."))
    scoring = argschema.fields.List(
        argschema.fields.Str,
        required=False,
        cli_as_single_argument=True,
        default=['roc_auc'],
        description=("evaluated metrics for the model. See "
                     "https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter"))  # noqa
    refit = argschema.fields.Str(
        required=False,
        default='roc_auc',
        description=("metric for refitting the model. See "
                     "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"))  # noqa

    @mm.post_load
    def validate_s3_or_input(self, data, **kwargs):
        for k in ['training_data', 'test_data']:
            if not data[k].startswith("s3://"):
                argschema.fields.files.validate_input_path(data[k])
            else:
                uri = urlparse(data[k])
                if not object_exists(uri.netloc, uri.path[1:]):
                    raise mm.ValidationError(f"{uri.geturl()} does not exist")
        return data


def train_classifier(training_data_path: str, scoring: List[str],
                     refit: str) -> GridSearchCV:
    """Performs k-fold cross-validated grid search logistic regression

    Parameters
    ----------
    training_data_path: str
        local path or s3 URI to training data in json format
    scoring: List[str]
        passed to GridSearchCV to specify tracked metrics
    refit: str
        passed to GridSearchCV to specify refit metric

    Returns
    -------
    clf: GridSearchCV
        the trained model

    """
    logger.info('Reading training data and extracting features.')
    training_data = json_load_local_or_s3(training_data_path)

    features = FeatureExtractor.from_list_of_dict(training_data).run()
    labels = [r['label'] for r in training_data]

    logger.info('Fitting model to data!')
    pipeline = feature_pipeline()
    model = LogisticRegression(penalty='elasticnet', solver='saga')
    pipeline.steps.append(('model', model))
    k_folds = KFold(n_splits=5)
    param_grid = {'model__l1_ratio': [0.25, 0.5, 0.75]}
    clf = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring,
                       cv=k_folds, refit=refit)
    logger.info(f"fitting model with {clf.get_params()}")
    clf.fit(features, labels)
    return clf


def mlflow_log_classifier(training_data_path: str,
                          test_data_path: str,
                          clf: GridSearchCV) -> str:
    """Logs a classifier with mlflow

    Parameters
    ----------
    training_data_path: str
        path or URI of the training data
    test_data_path: str
        path or URI of the test data
    clf: GridSeachCV
        a trained classifier

    Returns
    -------
    run_id: str
        the mlflow-assigned run_id

    """
    # log the run
    with mlflow.start_run() as mlrun:
        mlflow.set_tags({'training_data_path': training_data_path,
                         'param_grid': clf.param_grid})

        for k, v in clf.best_params_.items():
            mlflow.log_metric(k, v)
        for score_key in clf.scorer_.keys():
            keys = [f"split{i}_test_{score_key}"
                    for i in range(clf.n_splits_)]
            for k in keys:
                mlflow.log_metric(k, clf.cv_results_[k][clf.best_index_])

        # log and save fitted model
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_model_path = Path(temp_dir) / "trained_model.joblib"
            joblib.dump(clf.best_estimator_, tmp_model_path)
            mlflow.log_artifact(str(tmp_model_path))

        # run the model on test_data
        test_data = json_load_local_or_s3(test_data_path)
        features = FeatureExtractor.from_list_of_dict(test_data).run()
        y_true = [r['label'] for r in test_data]
        y_pred = clf.predict(features)

        mlflow.log_metric('test_accuracy', accuracy_score(y_true, y_pred))
        cmat = confusion_matrix(y_true, y_pred)
        for i in [0, 1]:
            for j in [0, 1]:
                mlflow.log_metric(f"count_{i}_{j}", int(cmat[i, j]))

        run_id = mlrun.info.run_id

    return run_id


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # train the classifier
        clf = train_classifier(
                training_data_path=self.args['training_data'],
                scoring=self.args['scoring'],
                refit=self.args['refit'])
        self.logger.info(
            f"Model fit, with best score {clf.best_score_} "
            f"and best parameters {clf.best_params_}.")

        # log the training
        run_id = mlflow_log_classifier(
                self.args['training_data'],
                self.args['test_data'],
                clf)
        self.logger.info(f"logged training to mlflow run {run_id}")


if __name__ == '__main__':  # pragma no cover
    trainer = ClassifierTrainer()
    trainer.train()
