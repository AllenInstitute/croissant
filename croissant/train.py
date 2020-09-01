from pathlib import Path
import joblib
import tempfile
import logging
import time
from functools import partial
from urllib.parse import urlparse
from typing import Type, List, Optional, Callable, Union, Iterable

import marshmallow as mm
import numpy as np
import mlflow
import mlflow.sklearn
import argschema
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, get_scorer
from sklearn.base import clone, BaseEstimator
import sklearn

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hpsklearn.components import _trees_hp_space

from croissant.features import FeatureExtractor, feature_pipeline
from croissant.utils import json_load_local_or_s3, object_exists

logger = logging.getLogger("TrainClassifier")


class KFoldTuner(object):
    """Tune a classifier with hyperopt on k-fold cross-validation.
    This class contains the methods for easily tuning classifier
    hyperparameters with an arbitrary pipeline. It will not tune the
    parameters in the rest of the pipeline, only the final estimator
    step (the classifier).

    The hyperopt-sklearn library exists to make tuning sklearn models
    with hyperopt, but currently they do not have enough preprocessing
    methods implemented for us to use without changing our pipeline.
    It may be useful to adopt it in the future.

    This will work with any binary classifier that implements the
    sklearn api (including xgboost for example). To create a tuner
    with a default parameter space to search, subclass this class.
    """
    def __init__(
            self, X: Iterable,
            y: Iterable,
            classifier: Type[BaseEstimator],
            param_space: dict,
            pipeline: Optional[Pipeline] = None,
            n_splits: int = 5,
            scorer: Union[str, Callable] = "roc_auc",
            seed: int = 42,
            opt_algo: Optional[Callable] = None):
        """
        Parameters
        ----------
        X: Iterable
            Feature vector. Input data to train the model.
        y: Iterable
            True labels for the data to train the model.
        classifier: Type[BaseEstimator]
            A scikit-learn classifier class. Should be able to be
            instantiated from params in the param space.
        param_space: dict
            The sampling space for parameter search. Should evaluate
            to a pyll graph. See
            http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
        pipeline: Pipeline
            An optional Pipeline. Will use to preprocess data, and
            append the classifier to the end of the Pipeline.
            Will not search for hyperparameters to tune in the Pipeline.
        n_splits: int, default=5
            Number of k-folds to use for computing score. Default=5
        scorer: str, Callable
            A str (see sklearn model evaluation documentation) or a
             scorer callable object / function with signature
            scorer(estimator, X, y) which should return only a single
            value.
        seed: int, default=42
            Random seed
        opt_algo: Callable
            Optimization algorithm to use. Either tpe.rand.suggest or
            tpe.suggest. If not specified will default to tpe.suggest
        """
        self.param_space = param_space
        self.X = X
        self.y = y
        self.classifier = classifier
        self.pipeline = pipeline
        self.kfolds = KFold(
            n_splits=n_splits, random_state=seed, shuffle=True)
        self.scorer = scorer
        self.opt_algo = opt_algo or tpe.suggest
        self.seed = seed
        self.trials = Trials()

    @staticmethod
    def _objective(
            params: dict,
            *,
            n_splits: int = 5,
            classifier: Type[BaseEstimator],
            pipeline: Optional[Pipeline] = None,
            X: Iterable, y: Iterable,
            scorer: Union[str, Callable],
            low_good: bool = True) -> dict:
        """
        Evaluate performance of a single model in a hyperopt trial.

        Parameters
        ----------
        params: dict
            Dictionary of parameters to be passed to the estimator.
        n_splits: int, default=5
            Number of k-folds to use for computing score. Default=5
        classifier: Type[BaseEstimator]
            A scikit-learn classifier class. Should be able to be
            instantiated from **params.
        pipeline: Pipeline
            An optional Pipeline. Will use to preprocess data, and
            append the classifier to the end of the Pipeline.
        X: Iterable
            Feature vector. Input data to train the model.
        y: Iterable
            True labels for the data to train the model.
        scorer: str, Callable
            A str (see sklearn model evaluation documentation) or a
            scorer callable object / function with signature
            scorer(estimator, X, y) which should return only a single
            value.
        low_good: bool
            Whether a low score is better than a high score.

        Returns
        -------
        Dictionary of results with the following keys:
            loss: The model's average score using `scorer` over
                  `nfolds` folds. If `low_good` is False, then return
                  the negative of the score.
            train_time_s: The time taken to train the model (seconds)
            status: hyperopt status report; OK if successfully completed
        """
        # To add a new classifier to the end of the pipeline, need to clone
        # existing pipeline; otherwise, classifiers will keep getting added
        # on to the pipeline since pipeline `append` updates the reference.
        start_time = time.time()
        estimator = classifier(**params)
        if pipeline is None:
            new_pipe = Pipeline(steps=[("classifier", estimator)])
        else:
            new_pipe = clone(pipeline)
            new_pipe.steps.append(("classifier", estimator))
        loss = np.mean(
            cross_val_score(new_pipe, X, y, scoring=scorer, cv=n_splits))
        if not low_good:
            # Hyperopt minimizes the optimization metric; if a high score is
            # good, return the negative
            loss = loss * -1
        # Include additional metadata for `trials` object (useful for tracking)
        result = {
            "loss": loss,
            "train_time_s": time.time() - start_time,
            "status": STATUS_OK,
            }
        return result

    def fmin(self, max_evals: int) -> dict:
        """Optimization function. Minimizes the objective metric.

        Parameters
        ----------
        max_evals: int
            Maximum number of trials to evaluate when searching for
            hyperparameters

        Returns
        -------
        Dictionary of parameters that produced the best metric on
        the objective function.
        """
        objective_fn = self._objective_fn(low_good=True)
        best_set = fmin(
            fn=objective_fn,
            space=self.param_space,
            algo=self.opt_algo,
            trials=self.trials,
            # Deprecated in numpy but required for hyperopt
            rstate=np.random.RandomState(self.seed),
            max_evals=max_evals
        )
        return space_eval(self.param_space, best_set)

    def _objective_fn(self, low_good: bool) -> Callable:
        """Build the objective function, using partial to 'fill in' the
        required arguments so all it takes are the parameters from
        hyperopt. Used for `fmin` and `fmax`.
        """
        objective_fn = partial(
            self._objective,
            n_splits=self.kfolds, classifier=self.classifier,
            pipeline=self.pipeline, X=self.X, y=self.y, scorer=self.scorer,
            low_good=low_good)
        return objective_fn

    def fmax(self, max_evals: int) -> dict:
        """ Maximization version of fmin. Use when the scoring metric
        should be maximized. Provided for convenience, since all
        sklearn scorers follow the convention that higher return values
        are better than lower return values. This way the user does not
        need to remember to negate the score to use `fmin`.

        Parameters
        ----------
        max_evals: int
            Maximum number of trials to evaluate when searching for
            hyperparameters

        Returns
        -------
        Dictionary of parameters that produced the best metric on
        the objective function.
        """
        objective_fn = self._objective_fn(low_good=False)
        best_set = fmin(
            fn=objective_fn,
            space=self.param_space,
            algo=self.opt_algo,
            trials=self.trials,
            # Deprecated in numpy but required for hyperopt
            rstate=np.random.RandomState(self.seed),
            max_evals=max_evals
        )
        return space_eval(self.param_space, best_set)


class RandomForestTuner(KFoldTuner):
    """Special case of KFoldTuner with RandomForestClassifier and a
    default parameter space option.
    See KFoldTuner for more information."""
    def __init__(
            self,
            X: Iterable,
            y: Iterable,
            pipeline: Optional[Pipeline],
            param_space: Optional[dict] = None,
            opt_algo: Optional[Callable] = None,
            n_splits: int = 5,
            scorer: Union[str, Callable] = "roc_auc",
            seed: int = 42,
            n_jobs: int = 1):
        """
        Parameters
        ----------
        X: Iterable
            Feature vector. Input data to train the model.
        y: Iterable
            True labels for the data to train the model.
        classifier: Type[BaseEstimator]
            A scikit-learn classifier class. Should be able to be
            instantiated from params in the param space.
        pipeline: Pipeline
            An optional Pipeline. Will use to preprocess data, and
            append the classifier to the end of the Pipeline.
            Will not search for hyperparameters to tune in the Pipeline.
        param_space: dict
            The sampling space for parameter search. Should evaluate
            to a pyll graph. See
            http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
        n_splits: int, default=5
            Number of k-folds to use for computing score. Default=5
        scorer: str, Callable
            A str (see sklearn model evaluation documentation) or a
             scorer callable object / function with signature
            scorer(estimator, X, y) which should return only a single
            value.
        seed: int, default=42
            Random seed
        opt_algo: Callable
            Optimization algorithm to use. Either tpe.rand.suggest or
            tpe.suggest. If not specified will default to tpe.suggest
        """
        # Chose not to use hyperopt-sklearn for the main classifier
        # because of the limitations of preprocessing methods currently
        # implemented. Going to utilize the defaults for now though.
        default_space = _trees_hp_space(
            lambda x: x,    # for the naming function; just use the param name
            n_jobs=n_jobs,
            random_state=seed)
        space = param_space or default_space
        super().__init__(
            X, y, RandomForestClassifier, space, opt_algo=opt_algo,
            pipeline=pipeline, n_splits=n_splits, scorer=scorer, seed=seed)


class LogisticRegressionTuner(KFoldTuner):
    """Special case of KFoldTuner with LogisticRegression and a
        default parameter space option.
        See KFoldTuner for more information."""
    def __init__(
            self,
            X: Iterable,
            y: Iterable,
            pipeline: Optional[Pipeline],
            param_space: Optional[dict] = None,
            opt_algo: Optional[Callable] = None,
            n_splits: int = 5,
            scorer: Union[str, Callable] = "roc_auc",
            seed: int = 42,
            n_jobs: int = 1):
        """
        Parameters
        ----------
        X: Iterable
            Feature vector. Input data to train the model.
        y: Iterable
            True labels for the data to train the model.
        classifier: Type[BaseEstimator]
            A scikit-learn classifier class. Should be able to be
            instantiated from params in the param space.
        pipeline: Pipeline
            An optional Pipeline. Will use to preprocess data, and
            append the classifier to the end of the Pipeline.
            Will not search for hyperparameters to tune in the Pipeline.
        param_space: dict
            The sampling space for parameter search. Should evaluate
            to a pyll graph. See
            http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
        n_splits: int, default=5
            Number of k-folds to use for computing score. Default=5
        scorer: str, Callable
            A str (see sklearn model evaluation documentation) or a
             scorer callable object / function with signature
            scorer(estimator, X, y) which should return only a single
            value.
        seed: int, default=42
            Random seed
        opt_algo: Callable
            Optimization algorithm to use. Either tpe.rand.suggest or
            tpe.suggest. If not specified will default to tpe.suggest
        """
        default_space = {
            "C": hp.choice("C", [100, 10, 1.0, 0.1, 0.01]),
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": hp.choice(
                "l1_ratio", [0, 0, 0, 1, 1, 1, 0.5, 0.2, 0.8])
            }
        space = param_space or default_space
        super().__init__(X, y, LogisticRegression, space, opt_algo=opt_algo,
                         pipeline=pipeline, n_splits=n_splits, scorer=scorer,
                         seed=seed)


def _binary_confusion_dict(
        y_true: Iterable, y_pred: Iterable, prefix: str = "") -> dict:
    """
    Helper function to unpack a confusion matrix from binary classifier
    results. Useful for logging purposes as it's easier to consume
    than an array.

    Parameters
    ----------
    y_true: Iterable
        Iterable of true labels for data (binary)
    y_pred: Iterable
        Iterable of predicted labels from a classifier (binary)
    prefix: Iterable, default=""
        Prefix for the keys in the returned dictionary

    Returns
    -------
    dict
        Dictionary with the following keys (prefixed with `prefix`):
            TN: true negative count
            FP: false positive count
            FN: false negative count
            TP: true positive count
    """
    if len(set(y_true)) > 2:
        raise ValueError(
            "Must have binary labels for confusion matrix dictionary.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Casting to int to avoid numpy types (don't play nice with postgres)
    return {f"{prefix}TN": int(tn), f"{prefix}FP": int(fp),
            f"{prefix}FN": int(fn), f"{prefix}TP": int(tp)}


def train_classifier(model: str,
                     training_data_path: str,
                     test_data_path: str,
                     scorer: Union[str, Callable],
                     max_iter: int,
                     optimizer: str,
                     test_metrics: Optional[List[str]] = None,
                     n_folds: int = 5,
                     seed: int = 42,
                     refit: bool = True,
                     drop_cols: List[str] = None):
    """Tunes and trains a model using hyperopt to optimize
    hyperparameters. Internally uses k-fold cross validation to
    compute optimization metrics. Uses `feature_pipeline` to
    preprocess data. The trained and tuned classifier is appended
    onto the pipeline as the final step.

    Parameters
    ----------
    model: str
        The model algorithm to use. One of "RandomForestClasifier"
        or "LogisticRegression". See `TrainingSchema.MODEL_CHOICES`.
    training_data_path: str
        local path or s3 URI to training data in json format
    test_data_path: str
        local path or s3 URI to training data in json format
    scorer: str
        A str (see sklearn model evaluation documentation).
        Will optimize for this scorer.
    max_iter: int
        Maximum number of iterations to evaluate hyperparameters
    optimizer: str
        Optimizer to use. One of "rand" or "suggest". See
        `TrainingSchema.OPTIMIZER_CHOICES`.
    test_metrics: List of str (or Callable)
        List of scorers to report metrics for model's performance on
        the test data set. See `scorer`.
    n_folds: int, default=5
        Number of folds over which to compute training metrics during
        hyperparameter optimization.
    seed: int, default=42
        Random seed
    refit: bool, default=True
        Whether to refit the model on the train and test set combined

    Returns
    -------
    4-tuple of:
        Pipeline
            the trained model, in a Pipeline object
        Optimized metric score (cross-validated)
            A tuple of (<scorer_name>, <scores>). The scores are for
            each fold in the training data.
        Test metrics
            A dictionary of score_name: score_value. These scores
            are computed on the test set only.
        confusion matrix
            A confusion matrix in dictionary format with the following
            keys: "TN", "TP", "FN", "FP" (corresponding to
            "true negative", "true positive", "false negative",
            "false positive", respectively).

    Notes
    -----
    When training a model for inference in the Allen Institute's production
    ophys pipeline, the data fed to FeatureExtractor should match between
    here and the production pipeline inference using the model:
    https://github.com/AllenInstitute/ophys_etl_pipelines/blob/37a03ec8d944b688c75da73e201824627d7f7df9/src/ophys_etl/transforms/classification.py#L426-L439  # noqa
    One consideration is that the production inference currently downsamples
    traces from 31Hz to 4Hz.
    Another consideration is the format. This training is written to use
    the FeatureExtractor.from_list_of_dict() method, while production
    inference uses the default constructor.

    """
    logger.info("Reading training data and extracting features.")
    training_data = json_load_local_or_s3(training_data_path)
    test_data = json_load_local_or_s3(test_data_path)
    features = FeatureExtractor.from_list_of_dict(training_data).run()
    test_features = FeatureExtractor.from_list_of_dict(test_data).run()
    labels = [r["label"] for r in training_data]
    test_labels = [r["label"] for r in test_data]

    # Instantiate model and (static) preprocessing pipeline
    pipeline = feature_pipeline(drop_cols=drop_cols)
    tuner_cls = TrainingSchema.MODEL_CHOICES[model]
    optimizer = TrainingSchema.OPTIMIZER_CHOICES[optimizer]
    tuner = tuner_cls(features, labels, pipeline, n_splits=n_folds,
                      scorer=scorer, opt_algo=optimizer, seed=seed)

    # Tune with CV and fit to training data
    logger.info(f"Fitting {model} to data (n={len(features)}).")
    start_time = time.time()
    logger.info(f"Optimizing '{scorer}' over {max_iter} iterations...")
    best_params = tuner.fmax(max_iter)
    end_time = time.time()
    logger.info(f"Considered {max_iter} trials over "
                f"{end_time-start_time} seconds. "
                f"Best Score: {-tuner.trials.best_trial['result']['loss']}\n"
                f"Best Params: {best_params}")
    clf = tuner.classifier(**best_params)
    pipeline.steps.append(("classifier", clf))
    logger.info(f"Fitting {model} model to training data with {best_params}")
    pipeline.fit(features, labels)

    # Compute cross-validation score for optimization metric
    # Note that this metric will be an optimistic estimate of performance,
    # since we are not using nested cross-validation
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    opt_score = (scorer, cross_val_score(
        pipeline, features, labels, scoring=scorer, cv=cv))

    # Compute metrics for test data
    # Default to just computing the optimization metric on test data
    test_metrics = test_metrics or [scorer]
    test_metrics = {
        f"test_{metric}":
            get_scorer(metric)(pipeline, test_features, test_labels)
        for metric in test_metrics}

    # Compute confusion matrix for arbitrary metric calculation
    cmat = _binary_confusion_dict(
        test_labels, pipeline.predict(test_features), "test_")

    # Refit on full data if applicable
    if refit:
        full_features = pd.concat([features, test_features], axis=0)
        full_labels = labels + test_labels
        pipeline.fit(full_features, full_labels)
    return pipeline, opt_score, test_metrics, cmat


def mlflow_log_classifier(training_data_path: str,
                          test_data_path: str,
                          clf: Pipeline,
                          cv_scores: tuple,
                          metrics: Optional[dict] = None,
                          confusion_matrix: Optional[dict] = None,
                          tags: Optional[dict] = None) -> str:
    """Logs a classifier with mlflow.

    Parameters
    ----------
    training_data_path: str
        path or URI of the training data
    test_data_path: str
        path or URI of the test data
    score: tuple
        Tuple of (<score name>, <scores per fold>) for optimization
        metric (training data). Encouraged by scikit-learn's model
        persistence page to ensure reproducibility.
    clf: Pipeline
        a Pipeline containing a trained classifier as the final step.
    metrics: dict
        Optional dictionary of {<score name>, <value>} for scored
        metrics on test data.
        Example: {"accuracy": 0.95, "precision": 0.89}.
    confusion_matrix: dict
        An optional confusion matrix for classifier results. If
        included, will allow computing arbitrary metrics from the
        model performance log. The keys should be all of the following:
        "TN", "TP", "FN", "FP" (true negative, true positive, etc.).
        The values should be the counts for each of these possible
        keys (not normalized).
    tags: dict (default=None)
        An optional dictionary of tags. By default will tag the
        paths to the train and test data, and the scikit-learn version
        used to train the model.
    Returns
    -------
    run_id: str
        the mlflow-assigned run_id
    """
    # log the run
    mlrun = mlflow.start_run()
    tags = tags or {}
    mlflow.set_tags({"training_data_path": training_data_path,
                     "test_data_path": test_data_path,
                     "sklearn_version": sklearn.__version__,
                     **tags})
    # Log the CV scores for optimization metric
    # This is important for reproducibility
    # See https://scikit-learn.org/stable/modules/model_persistence.html
    for ix, score in enumerate(cv_scores[1]):
        mlflow.log_metric(f"split{ix}_train_{cv_scores[0]}", score)

    # Now log the test scores if given
    if metrics:
        mlflow.log_metrics(metrics)

    # log and save fitted model
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_model_path = Path(temp_dir) / "trained_model.joblib"
        joblib.dump(clf, tmp_model_path)
        mlflow.log_artifact(str(tmp_model_path))

    # log model parameters
    mlflow.log_params(clf.steps[-1][1].get_params())

    # Log confusion matrix
    if confusion_matrix is not None:
        mlflow.log_metrics(confusion_matrix)
    run_id = mlrun.info.run_id
    mlflow.end_run()
    return run_id


class TrainingSchema(argschema.ArgSchema):
    """Schema for parsing arguments to the main module behavior."""

    MODEL_CHOICES = {"RandomForestClassifier": RandomForestTuner,
                     "LogisticRegression": LogisticRegressionTuner}
    OPTIMIZER_CHOICES = {"tpe": tpe.suggest, "rand": tpe.rand.suggest}

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
    scorer = argschema.fields.Str(
        required=False,
        missing="roc_auc",
        description=("Optimization metric for the model. All scorers follow "
                     "the convention that higher return values are better "
                     "than lower return values. See "
                     "https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter"))  # noqa
    reported_metrics = argschema.fields.List(
        argschema.fields.Str,
        required=False,
        missing=["roc_auc"],
        cli_as_single_argument=True,
        description=("Metrics to report to mlflow, in addition to the scorer "
                     "(optimization metric)."))
    model = argschema.fields.Str(
        required=False,
        missing="RandomForestClassifier",
        validate=mm.validate.OneOf(list(MODEL_CHOICES)),
        description=(f"Choice of model. Must be one of {list(MODEL_CHOICES)}")
    )
    max_iter = argschema.fields.Int(
        required=False,
        missing=100,
        validate=lambda x: x > 0,
        description=("Number of trials to conduct hyperparameter tuning.")
    )
    refit = argschema.fields.Bool(
        required=False,
        missing=True,
        description="Whether to refit model on full data (train+test set)."
    )
    seed = argschema.fields.Int(
        required=False,
        missing=42,
        description=("Random seed to reproduce folds, etc.")
    )
    optimizer = argschema.fields.Str(
        required=False,
        missing="tpe",
        validate=mm.validate.OneOf(["tpe", "rand"]),
        description=("Optimizer to use for hyperparamter tuning. One of "
                     "'rand' (random search) or 'tpe' (Bayesian optimization)")
    )
    n_folds = argschema.fields.Int(
        required=False,
        missing=5,
        validate=lambda x: x > 0,
        description="Number of folds for k-fold cross-validation."
    )
    drop_cols = argschema.fields.List(
        argschema.fields.Str,
        missing=None,
        cli_as_single_argument=True,
        description=("Feature columns to drop from the input data.")
    )

    @mm.post_load
    def validate_s3_or_input(self, data, **kwargs):
        for k in ["training_data", "test_data"]:
            if not data[k].startswith("s3://"):
                argschema.fields.files.validate_input_path(data[k])
            else:
                uri = urlparse(data[k])
                if not object_exists(uri.netloc, uri.path[1:]):
                    raise mm.ValidationError(f"{uri.geturl()} does not exist")
        return data


class ClassifierTrainer(argschema.ArgSchemaParser):
    """Main module entry point."""
    default_schema = TrainingSchema

    def train(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop("log_level"))

        tags = {"model": self.args["model"],
                "metric": self.args["scorer"],
                "seed": self.args["seed"],
                "n_folds": self.args["n_folds"],
                "max_iter": self.args["max_iter"],
                "optimizer": self.args["optimizer"],
                "refit": self.args["refit"]}

        # train the classifier
        clf, cv_scores, test_metrics, confusion_matrix = train_classifier(
                model=self.args["model"],
                training_data_path=self.args["training_data"],
                test_data_path=self.args["test_data"],
                scorer=self.args["scorer"],
                max_iter=self.args["max_iter"],
                optimizer=self.args["optimizer"],
                test_metrics=self.args["reported_metrics"],
                n_folds=self.args["n_folds"],
                seed=self.args["seed"],
                refit=self.args["refit"],
                drop_cols=self.args["drop_cols"])

        # log the training
        run_id = mlflow_log_classifier(
                self.args["training_data"],
                self.args["test_data"],
                clf,
                cv_scores,
                metrics=test_metrics,
                confusion_matrix=confusion_matrix,
                tags=tags
        )
        self.logger.info(f"logged training to mlflow run {run_id}")


if __name__ == "__main__":  # pragma no cover
    trainer = ClassifierTrainer()
    trainer.train()
