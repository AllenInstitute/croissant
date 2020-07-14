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
These instructions also assume that the user has the [AWS Command Line Interface](https://aws.amazon.com/cli/) installed and that it can find apprpriate credentials. We can then provide `mlflow` credentials via the AWS secret manager. This assumes the JSON processor `jq` is installed.
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
Not yet implemented. Coming soon.
