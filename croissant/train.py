from pathlib import Path
import json
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
import mlflow
import mlflow.sklearn
import argschema
from typing import Dict, Any

from croissant.schemas import TrainingSchema
from croissant.roi import RoiWithMetadata
from croissant.features import FeatureExtractor, feature_pipeline


logger = logging.getLogger('TrainClassifier')


def train_classifier(training_data_path: Path, output_dir: Path,
                     search_grid: Dict[str, Any]):
    """Performs k-fold cross-validated grid search logistic regression and
    logs to mlflow.

    Parameters
    ----------
    training_data_path: Path
        The path to the ROIs stored in json format for the classifier to train
        against
    output_dir: Path
        The path to the directory where the output files will be stored
    search_grid: Dict[str, Any]
        The parameters upon which to grid search across during parameter tuning

    Returns
    -------

    """
    # set tracker
    with mlflow.start_run():
        mlflow.set_tags({'training_data_path': training_data_path,
                         'search_grid': search_grid,
                         'output_dir': output_dir})

        logger.info("Extracting ROI data from manifest data!")
        with open(training_data_path, 'r') as fp:
            training_data = json.load(fp)
        roi_list = [RoiWithMetadata.from_dict(r) for r in training_data]
        rois = [r.roi for r in roi_list]
        dff_traces = [r.trace for r in roi_list]
        metadatas = [r.roi_meta for r in roi_list]
        labels = [r.label for r in roi_list]

        logger.info('Extracting features!')
        features = FeatureExtractor(rois=rois,
                                    dff_traces=dff_traces,
                                    metadata=metadatas).run()

        logger.info('Fitting model to data!')
        pipeline = feature_pipeline()
        model = LogisticRegression(penalty='elasticnet', solver='saga')
        pipeline.steps.append(('model', model))
        scorers = {'AUC': 'roc_auc'}
        k_folds = KFold(n_splits=5)
        clf = GridSearchCV(pipeline, param_grid=search_grid, scoring=scorers,
                           cv=k_folds, refit='AUC')
        logger.info(f"fitting model with {clf.get_params()}")
        clf.fit(features, labels)

        logger.info(
            f"Model fitted, the best score is {clf.best_score_} "
            f"and the best parameters are {clf.best_params_}.")

        logger.info("Logging classification metrics")
        cv_results_frame = pd.DataFrame.from_dict(clf.cv_results_)
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric('Best_Score', clf.best_score_)
        for score_key, score_id in scorers.items():
            mlflow.log_metric(f'Mean_{score_key}',
                              cv_results_frame[f'mean_test_{score_key}'].max())
            mlflow.log_metric(f'STD_{score_key}',
                              cv_results_frame[f'std_test_{score_key}'].max())

        # log and save fitted model
        logger.info("Logging and Saving Fitted Model")
        mlflow.sklearn.log_model(clf, "FittedModel")

        logger.info('Classifier Trained, Goodbye!')


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):
        logger.setLevel(self.args['log_level'])

        if self.args['search_grid_path']:
            with open(self.args['search_grid_path']) as open_grid:
                search_grid = json.load(open_grid)
        else:
            search_grid = {'model__l1_ratio': [0.25, 0.5, 0.75]}

        mlflow.set_tracking_uri(self.args['mlflow_tracking_uri'])
        mlflow.set_experiment(self.args['experiment_name'])

        train_classifier(training_data=Path(self.args['training_data']),
                         output_dir=Path(self.args['output_dir']),
                         search_grid=search_grid)


if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train()
