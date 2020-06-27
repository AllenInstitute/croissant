from pathlib import Path
import json
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
import mlflow
import mlflow.sklearn
import argschema
import joblib
import tempfile
from typing import Dict, Any, List

from croissant.schemas import TrainingSchema
from croissant.features import FeatureExtractor, feature_pipeline


logger = logging.getLogger('TrainClassifier')


def train_classifier(training_data_path: Path, param_grid: Dict[str, Any],
                     scoring: List[str], refit: str) -> GridSearchCV:
    """Performs k-fold cross-validated grid search logistic regression

    Parameters
    ----------
    training_data_path: Path
        path to training data in json format
    param_grid: Dict[str, Any]
        passed to GridSearchCV to specify parameter grid
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
    with open(training_data_path, 'r') as fp:
        training_data = json.load(fp)
    features = FeatureExtractor.from_list_of_dict(training_data).run()
    labels = [r['label'] for r in training_data]

    logger.info('Fitting model to data!')
    pipeline = feature_pipeline()
    model = LogisticRegression(penalty='elasticnet', solver='saga')
    pipeline.steps.append(('model', model))
    k_folds = KFold(n_splits=5)
    clf = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring,
                       cv=k_folds, refit=refit)
    logger.info(f"fitting model with {clf.get_params()}")
    clf.fit(features, labels)
    return clf


def mlflow_log_classifier(training_data_path: Path, clf: GridSearchCV) -> str:
    """Logs a classifier with mlflow

    Parameters
    ----------
    training_data_path: Path
        path of the training data
    clf: GridSeachCV
        a trained classifier

    Returns
    -------
    run_id: str
        the mlflow-assigned run_id

    """
    with mlflow.start_run() as mlrun:
        mlflow.set_tags({'training_data_path': training_data_path,
                         'param_grid': clf.param_grid})

        cv_results_frame = pd.DataFrame.from_dict(clf.cv_results_)
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric('Best_Score', clf.best_score_)
        for score_key in clf.scorer_.keys():
            mlflow.log_metric(f'Mean_{score_key}',
                              cv_results_frame[f'mean_test_{score_key}'].max())
            mlflow.log_metric(f'STD_{score_key}',
                              cv_results_frame[f'std_test_{score_key}'].max())

        # log and save fitted model
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_model_path = Path(temp_dir) / "trained_model.joblib"
            joblib.dump(clf.best_estimator_, tmp_model_path)
            mlflow.log_artifact(tmp_model_path)

        run_id = mlrun.info.run_id

    return run_id


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # train the classifier
        clf = train_classifier(training_data=Path(self.args['training_data']),
                               param_grid=self.args['param_grid'],
                               scoring=self.args['scoring'],
                               refit=self.args['refit'])
        self.logger.info(
            f"Model fit, with best score {clf.best_score_} "
            f"and best parameters {clf.best_params_}.")

        # log the training
        mlflow.set_tracking_uri(self.args['mlflow_tracking_uri'])
        exp = mlflow.get_experiment_by_name(self.args['experiment_name'])
        if not exp:
            mlflow.create_experiment(
                self.args['experiment_name'],
                artifact_location=self.args['artifact_uri'])
        mlflow.set_experiment(self.args['experiment_name'])
        run_id = mlflow_log_classifier(self.args['training_data'], clf)
        self.logger.info(f"logged training to mlflow run {run_id}")


if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train()
