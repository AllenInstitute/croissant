import argschema


"""
ROI:
{
    'coo_rows': List[Int] (row position for each roi pixel),
    'coo_cols': List[Int] (col position for each roi pixel),
    'coo_data': List[Float] (data for each roi pixel),
    'image_shape': Tuple[Int, Int] (size of the image the roi is in),
    'experiment_id': Int (id of the experiment the roi is from),
    'roi_id': Int (id value for the roi),
    'trace': List[Float] (extracted calcium fluorescence signal) ,
    'rig' (camera that took the images of the experiment): str,
    'depth' (camera imaging depth): int,
    'full_genotype' (combo of all reporter/driver and transgenic lines): str,
    'targeted_structure' (structure of brain being imaged): str,
    'label' (true of false/cell or not cell/0 or 1): Int
}
Training Data: List[ROI]
"""


class TrainingSchema(argschema.ArgSchema):
    experiment_name = argschema.fields.String(
        required=True,
        description="Experiment name (for organization in MLFlow)")
    training_data = argschema.fields.InputFile(
        required=True,
        description=("Input file in json format containing the ROIs in COO "
                     "format with which to build the classifier. "
                     "Training data supplied needs to conform to the "
                     "format list in croissant/croissant/schemas.py"))
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
