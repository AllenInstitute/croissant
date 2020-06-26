import argschema


class TrainingSchema(argschema.ArgSchema):
    experiment_name = argschema.fields.String(
        required=True,
        description="Experiment name (for organization in MLFlow)")
    training_data = argschema.fields.InputFile(
        required=True,
        description=("<stem>.json containing a list of dicts, where "
                     "each dict can be passed into "
                     "RoiWithMetaData.from_dict()."))
    output_dir = argschema.fields.OutputDir(
        required=False,
        default='.',
        description=("Where to save output from dev runs (tracking metrics "
                     "and artifacts). Ignored if not in dev environment. "
                     "Defaults to current directory."))
    search_grid_path = argschema.fields.InputFile(
        required=False,
        default=None,
        allow_none=True,
        description=("Path to a *.json file containing a "
                     "parameter grid (List[Dict[str: Any]]"))
    mlflow_tracking_uri = argschema.fields.String(
        required=False,
        allow_none=True,
        default=None,
        description=("passed to mlflow.set_tracking_uri(). If None mlflow "
                     "sets to./mlruns"))
