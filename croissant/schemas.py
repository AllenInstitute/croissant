import argschema
import marshmallow as mm
import json


class TrainingSchema(argschema.ArgSchema):
    experiment_name = argschema.fields.String(
        required=True,
        description="Experiment name (for organization in MLFlow)")
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
    param_grid_path = argschema.fields.InputFile(
        required=False,
        default=None,
        allow_none=True,
        description=("Path to a *.json file containing a "
                     "parameter grid (List[Dict[str: Any]]"))
    param_grid = argschema.fields.Dict(
        required=False,
        default={'model__l1_ratio': [0.25, 0.5, 0.75]},
        description=("passed to GridSearchCV as `param_grid`. If "
                     "`param_grid_path` is passed, `param_grid` is set "
                     "with the contents of that file."))
    mlflow_tracking_uri = argschema.fields.String(
        required=False,
        allow_none=True,
        default=None,
        description=("passed to mlflow.set_tracking_uri(). If None mlflow "
                     "sets to./mlruns. This sets the backend store as: "
                     "https://www.mlflow.org/docs/latest/tracking.html#backend-stores"))  # noqa
    artifact_uri = argschema.fields.Str(
        required=True,
        description=("will be used as the artifact store, as: "
                     "https://www.mlflow.org/docs/latest/tracking.html#artifact-stores"))  # noqa

    @mm.post_load
    def check_artifact_uri(self, data, **kwargs):
        if not data['artifact_uri'].startswith("s3://"):
            argschema.fields.files.validate_outpath(data['artifact_uri'])
        return data

    @mm.post_load
    def check_param_grid(self, data, **kwargs):
        if data['param_grid_path'] is not None:
            with open(data['param_grid_path'], 'r') as f:
                data['param_grid'] = json.load(f)
        return data
