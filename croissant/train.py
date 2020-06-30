from pathlib import Path
import json
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
import mlflow
import mlflow.sklearn
import argschema
import joblib
import tempfile
from typing import List

from croissant.features import FeatureExtractor, feature_pipeline


logger = logging.getLogger('TrainClassifier')


class TrainingSchema(argschema.ArgSchema):
    training_data = argschema.fields.InputFile(
        required=True,
        description=("<stem>.json containing a list of dicts, where "
                     "each dict can be passed into "
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


def train_classifier(training_data_path: Path, scoring: List[str],
                     refit: str) -> GridSearchCV:
    """Performs k-fold cross-validated grid search logistic regression

    Parameters
    ----------
    training_data_path: Path
        path to training data in json format
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
    param_grid = {'model__l1_ratio': [0.25, 0.5, 0.75]}
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
    # log the run
    with mlflow.start_run() as mlrun:
        mlflow.set_tags({'training_data_path': training_data_path,
                         'param_grid': clf.param_grid})

        for k, v in clf.best_params:
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
            mlflow.log_artifact(tmp_model_path)

        run_id = mlrun.info.run_id

    return run_id


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # train the classifier
        clf = train_classifier(
                training_data_path=Path(self.args['training_data']),
                scoring=self.args['scoring'],
                refit=self.args['refit'])
        self.logger.info(
            f"Model fit, with best score {clf.best_score_} "
            f"and best parameters {clf.best_params_}.")

        # log the training
        run_id = mlflow_log_classifier(self.args['training_data'], clf)
        self.logger.info(f"logged training to mlflow run {run_id}")


if __name__ == '__main__':  # pragma no cover
    trainer = ClassifierTrainer()
    trainer.train()
