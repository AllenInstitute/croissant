from pathlib import Path
import json
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, GridSearchCV)
import pandas as pd
import mlflow
import mlflow.sklearn
import argschema

from croissant.schemas import (TrainingSchema)
from croissant.features import (Roi, RoiMetadata, FeatureExtractor,
                                feature_pipeline)


logger = logging.getLogger('TrainClassifier')


def train_classifier(environment: str, training_data: Path, output_dir: Path,
                     search_grid: dict):
    """
    Trains a classifier using the provided parameters defined by the user.
    The classifier takes ROI data as input and transform it into features
    that are then trained upon to fit data to model. Cross validation is
    performed according to k fold validation with 5 folds Tags, artifacts,
    metrics, and generated model are logged via MlFlow.
    Parameters
    ----------
    environment: str
        Dev, Test, Production - indicates where the training is being run.
        Dev will keep output files in specified output directory while Test
        and Production will remove these.
    training_data: Path
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

        # tag inputs with mlflow
        mlflow.set_tags({'training_data': training_data,
                         'search_grid': search_grid,
                         'output_dir': output_dir})

        # load the data, assume json format
        with open(training_data, 'r') as open_training:
            training_data_loaded = json.load(open_training)
            logger.info("Loaded ROI data from manifest.")

        rois = []
        dff_traces = []
        metadatas = []
        labels = []

        # extract data
        logger.info("Extracting ROI data from manifest data")
        for roi_data in training_data_loaded:
            dff_traces.append(roi_data['trace'])
            roi = Roi(roi_id=roi_data['roi_id'], coo_cols=roi_data['coo_cols'],
                      coo_rows=roi_data['coo_rows'],
                      coo_data=roi_data['coo_data'],
                      image_shape=roi_data['image_shape'])
            roi_meta = RoiMetadata(depth=roi_data['depth'],
                                   full_genotype=roi_data['full_genotype'],
                                   targeted_structure=roi_data['targeted_'
                                                               'structure'],
                                   rig=roi_data['rig'])
            labels.append(roi_data['label'])
            rois.append(roi)
            metadatas.append(roi_meta)
        logger.info("Extracted all ROI data and formatted for feature "
                    "extraction.")

        logger.info('Extracting features!')
        features = FeatureExtractor(rois=rois,
                                    dff_traces=dff_traces,
                                    metadata=metadatas).run()
        logger.info('Feature extraction complete!')

        # Fitting model
        logger.info('Fitting model to data!')
        pipeline = feature_pipeline()
        model = LogisticRegression(penalty='elasticnet', solver='saga')
        pipeline.steps.append(('model', model))

        # set the scorers
        scorers = {'AUC': 'roc_auc'}

        logger.info("CV strategy: K_fold, n_splits: 5, grid search will "
                    "occur with the selected cross validation strategy "
                    "on the specified parameter grid.")
        # grid search with cross validation
        k_folds = KFold(n_splits=5)
        gs = GridSearchCV(pipeline, param_grid=search_grid,
                          scoring=scorers, cv=k_folds,
                          refit='AUC')
        fitted_model = gs.fit(features, labels)
        cv_results_frame = pd.DataFrame.from_dict(gs.cv_results_)

        logger.info(
            f"Model fitted, the best score is {fitted_model.best_score_} "
            f"and the best parameters are {fitted_model.best_params_}.")

        logger.info("Logging classification metrics")
        # Only log the best performance with the best params, the
        # rest will be saved as an artifact in a pd Dataframe
        mlflow.log_params(fitted_model.best_params_)
        mlflow.log_metric('Best_Score', fitted_model.best_score_)
        for score_key, score_id in scorers.items():
            mlflow.log_metric(f'Mean_{score_key}',
                              cv_results_frame[f'mean_test_{score_key}'].max())
            mlflow.log_metric(f'STD_{score_key}',
                              cv_results_frame[f'std_test_{score_key}'].max())

        # log and save fitted model
        logger.info("Logging and Saving Fitted Model")
        mlflow.sklearn.log_model(fitted_model, "FittedModel")

        logger.info('Classifier Trained, Goodbye!')


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):

        # set up logger
        logger.setLevel(self.args['log_level'])

        # prepare args for handoff
        self.args['training_data'] = Path(self.args['training_data'])
        self.args['output_dir'] = Path(self.args['output_dir'])

        # Parse the search grid
        search_grid_path = self.args['search_grid_path']
        if search_grid_path:
            with open(search_grid_path) as open_grid:
                search_grid = json.load(open_grid)
        else:
            search_grid = {'model__l1_ratio': [0.25, 0.5, 0.75]}

        # Set up mlflow tracking
        mlflow_tracking_uri = self.args['mlflow_tracking_uri']
        experiment_name = self.args['experiment_name']
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        train_classifier(environment=self.args['environment'],
                         training_data=self.args['training_data'],
                         output_dir=self.args['output_dir'],
                         search_grid=search_grid)


if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train()
