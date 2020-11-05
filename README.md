[![CircleCI](https://circleci.com/gh/AllenInstitute/croissant.svg?style=svg)](https://circleci.com/gh/AllenInstitute/croissant)
[![codecov](https://codecov.io/gh/AllenInstitute/croissant/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenInstitute/croissant)

# croissant
The goal of this repo is to create the infrastructure capable of
creating a classifier to classify segmented ROIs in two photon
microscopy video into Cell/Not Cell. It is currently in active
development. The community is welcome to use this repository as an
infrastructure to create classifiers for two photon microscopy video.

## Level of support
We are not currently supporting this code, but simply releasing it to
the community AS IS. We are not able to provide any guarantees of
support. The community may submit issues, but you should not expect an
active response.

## Contributing
This tool is important for internal use at the Allen Institute. Because
it's designed for internal needs, we are not anticipating external
contributions. Pull requests may not be accepted if they conflict with
our existing plans.

## Training Examples
This repo uses [`mlflow`](https://www.mlflow.org/) to log model training. `mlflow` supports a number of backends.

### Local file mlflow backend, local execution
specify to `mlflow` the local file backend.
```bash
export MLFLOW_TRACKING_URI=${HOME}/mlflow_example/tracking
```
also specify a location for the artifacts, and an experiment name
```bash
export ARTIFACT_URI=${HOME}/mlflow_example/artifacts
export MYEXPERIMENT=example_experiment
```
create an experiment
```bash
mlflow experiments create \
    --experiment-name ${MYEXPERIMENT} \
    --artifact-location ${ARTIFACT_URI}
```
An `mlflow` experiment holds a collection of runs. It needs to be created only once, before the first run.
Perform a run, driven by the `MLproject` file in this repo.
```bash
mlflow run ${HOME}/croissant/  \
    --backend local \
    --experiment-name ${MYEXPERIMENT} \
    -P training_data=${HOME}/data/training_data.json \
    -P test_data=${HOME}/data/testing_data.json
    -P log_level=INFO
``` 
The local mlflow tracking server can be interfaced with a UI:
```bash
mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI}
```

### AWS-hosted mlflow backend, local processing
establish a vpn connection to the AWS VPC. This example is using `openvpn` on linux and other examples can be found [here](https://docs.aws.amazon.com/vpn/latest/clientvpn-user/connect.html). Probably one can fix permissions so `sudo` is not necessary. The contents of the `openvpn` config file is provided to the user by an IAM adminsitrator.
```bash
sudo openvpn --config .config/openvpn/mlflow_vpn_config.ovpn
```
These instructions also assume that the user has the [AWS Command Line Interface](https://aws.amazon.com/cli/) installed and that it can find appropriate credentials. We can then provide `mlflow` credentials via the AWS secret manager. This assumes the JSON processor `jq` is installed.
```
SECRET_NAME=mlflow-stack-prod-MLFlowDbSecret
secret=$(aws secretsmanager get-secret-value --secret-id $SECRET_NAME --query SecretString --output text) &&  USER=$(echo $secret | jq .username -r) &&   PASSWORD=$(echo $secret | jq .password -r) &&   HOST=$(echo $secret | jq .host -r) &&   PORT=$(echo $secret | jq .port -r) &&   DBNAME=$(echo $secret | jq .dbname -r)
```
The environment variable for the remote `mlflow` postgres backend is:
```bash
export MLFLOW_TRACKING_URI=postgresql://$USER:$PASSWORD@$HOST:$PORT/$DBNAME
```
At this point, you can once again launch the `mlflow` UI, as shown in the local example, with this new ENV variable value.
To create an experiment, again, we need to specify where those artifacts are to be stored, and a name for the experiment:
```bash
export ARTIFACT_URI=s3://<bucket>/<prefix>
export MYEXPERIMENT=example_aws_experiment
```
The create experiment and run commands are the same as above. `--backend local` is telling `mlflow` to run the processing on the local machine. This repo also supports S3 URIs for the training and test data sources.

### AWS-hosted mlflow backend, AWS-hosted processing
The [AWS Cloudformation](https://aws.amazon.com/cloudformation/) stack described in `deploy/cloudformation/mlflow-db-template.yml` establishes resources for [AWS Fargate](https://aws.amazon.com/fargate/) tasks. These are mlflow training tasks run in serverless containers via the client command line. The image for the containers is auto-built and deployed to [DockerHub](https://hub.docker.com/r/alleninstitutepika/croissant) with images tagged by git commit hash. The cloudformation stack can be refreshed for a new image tag like:
```
aws cloudformation deploy \
  --template-file ./deploy/cloudformation/mlflow-db-template.yml \
  --stack-name mlflow-test-stack \
  --parameter-overrides ImageUri=docker.io/alleninstitutepika/croissant:da75ca5fdac9f7e857662fd48e0776ed3628dbeb \
  --capabilities CAPABILITY_NAMED_IAM
```

Launching a Fargate task requires knowledge of various AWS resource names, the Mlflow backend URI, and an appropriate Mlflow experiment. The `croissant.online_training_helper` is here to help:
```
$ python -m croissant.online_training_helper --stack mlflow-test-stack
INFO:root:ENV variables to use stack mlflow-test-stack

export ONLINE_TRAINING_CLUSTER=mlflow-test-stack-fargate-cluster
export ONLINE_TRAINING_TASK=arn:aws:ecs:us-west-2:606907419058:task-definition/container-task:14
export ONLINE_TRAINING_CONTAINER=croissant-container
export ONLINE_TRAINING_SUBNET=<subnet Id>
export ONLINE_TRAINING_SECURITY=<security group Id>
export MLFLOW_TRACKING_URI=postgresql://<user>:<password>@<host>:<port>/<dbname>

INFO:root:Available experiments with s3 artifact stores:

export MLFLOW_EXPERIMENT=example_aws_experiment
```
The current production stack is named: `croissant-mlflow-prod-stack`

The above are suggestions made by the helper. Taking the suggestions and executing these export statements makes running the online training tasks quite simple (though tedious arg-by-arg specification of all the resources is still possible):
```
python -m croissant.online_training \
  --output_json ./output.json \
  --training_args.training_data s3://prod.slapp.alleninstitute.org/merged_2line_2project/training_data.json \
  --training_args.test_data s3://prod.slapp.alleninstitute.org/merged_2line_2project/testing_data.json \
  --training_args.log_level INFO
```

A json can also be used as input to the module to make the launch command easier. It can be helpful when
specifying more complex configurations, such as a list of dropped columns or desired metrics. For example:

```
{
  "training_args": {
    "training_data": "s3://prod.slapp.alleninstitute.org/merged_2line_2project/training_data.json",
    "test_data": "s3://prod.slapp.alleninstitute.org/merged_2line_2project/testing_data.json",
    "log_level": "INFO",
    "scorer": "roc_auc",
    "reported_metrics": ["roc_auc", "average_precision", "recall", "precision", "accuracy"],
    "drop_cols": ["full_genotype", "targeted_structure", "_feat_last_tenth_trace_skew", "_feat_simple_n_spikes",    "_feat_roi_data_skew", "_feat_roi_data_std"],
    "seed": 42,
    "refit": true,
    "max_iter": 2000
  }
}
```
See `croissant.train.py` for a full description of the available training arguments.

Once launched, these tasks can be tracked n the AWS console `ECS -> Cluster -> Tasks` where the cluster name is based on the cloudformation stack name. For example `mlflow-test-stack-fargate-cluster `. The logs of the job can be found from `Cloudwatch -> Logs -> LogGroups` and the log group will also be named based on the stack name, for example `mlflow-test-stack-ecs `.

The container image that is run above is specified at cloudformation stack creation time, or via a stack deployment changeset application. We can also dynamically supply a container image on the command line. This can be helpful for quicker development iteration. In this case, a new task definition for ECS will be created and run, rather than the existing stack-defined task defintion. The additional argument is `container_image`:
```
python -m croissant.online_training \
  --output_json ./output.json \
  --training_args.training_data s3://prod.slapp.alleninstitute.org/merged_2line_2project/training_data.json \
  --training_args.test_data s3://prod.slapp.alleninstitute.org/merged_2line_2project/testing_data.json \
  --training_args.log_level INFO \
  --container_image docker.io/alleninstitutepika/croissant:7ca6c6968866ad1b545df5c8dc92544809864413
```
