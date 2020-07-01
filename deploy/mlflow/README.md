# Monitoring Runs with MLFlow

We have provided a docker container to make it easy to run the MLFlow UI directly,
without having to manage any credentials or dependencies (outside of your AWS IAM role).
You can pull directly from ECR, or build it yourself from source.

## Retrieve container from repository
First you'll need to authenticate Docker to the ECR registry. 
See instructions [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Registries.html#registry_auth).

To pull from ECR, check the images in the repository:

```
aws ecr describe-images --repository-name croissant-mlflow
```

To pull the `latest` tag, use the following command (replace the bracketed text
with your AWS account id).

```
docker pull <REPLACE_ME_AWS_ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/croissant-mlflow:latest
```

If you don't know how to access your account ID, you can retrieve it with the CLI.

```
echo $(aws sts get-caller-identity | jq .Account)
```

You can also just run `aws sts get-caller-identity` and use the value from the "Account" key in the output
json if you don't want to install jq.

## Build container from source
If you don't want to pull from ECR, you can build it yourself. 
First, clone this repository. Then, from the root repo directory:

```
docker build . -f deploy/mlflow/mlflow.Dockerfile -t croissant-mlflow:latest
```

## Run MLFlow UI
The MLFlow UI can be hosted on your remote machine, or you can start it
on an EC2 instance inside the same VPC as the tracking database 
(ensure the instance is using a security group that allows inbound access to the database). 
If you are hosting it outside of the VPC, you must use a VPN to get access to the database.

### Remote machines: Start VPN
We use Aurora Serverless Postgres database to store our model information.
There is no public access to this database; you must be in the same VPC.
The MLFlow UI will only work if you are connected to the VPC using the Client VPN.

Contact <kat.schelonka@alleninstitute.org> for the information required to 
authenticate with and configure the VPN.

After receiving your configuration file, download an OpenVPN client appropriate
for your operating system (e.g. TunnelBlick for MacOs). Follow the instructions
to import the configuration.

You may receive a warning that your apparent IP address has not changed. If
you are able to connect to the MLFlow UI, you can safely ignore this.

### Run the container
To run the MLFlow UI from the docker container, you need to bind the default
port (5000) on the container to a port on our machine so that you can access
it through the web browser. You also must mount your ~/.aws directory volume
so that the AWS CLI can use your credentials and settings. The & at the end
runs the process in the background. Finally, you'll have to pass the name of
the secret in AWS Secrets Manager that contains the credentials to access the
MLFlow Database (running on postgres).

```
docker run \
    -p 5000:5000 \
    -v ~/.aws:/root/.aws \
    --env SECRET_NAME=<name of secret in secrets manager to access tracking DB> \
    --rm \
    croissant-mlflow:latest \
    &
```
Open your browser and navigate to localhost:5000. You should see the MLFlow UI.

## Exiting the UI
Stop the container when you are finished. The `--rm` flag should automatically
clean up.  Use `docker ps` to see your running docker processes, then
`docker stop <name>` to stop the container.

If you are using a VPN to connect to resources in a VPC, make sure to exit it.