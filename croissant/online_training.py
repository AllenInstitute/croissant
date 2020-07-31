import boto3
import os
import argschema
import marshmallow as mm
import json


class OnlineTrainingException(Exception):
    pass


class OnlineTrainingSchema(argschema.ArgSchema):
    cluster = argschema.fields.Str(
        required=False,
        default=None,
        missing=None,
        allow_none=True,
        description=("The cluster name. If not provided will attempt to get "
                     "environment variable ONLINE_TRAINING_CLUSTER. See boto3 "
                     "docs for run_task arg `cluster`"))
    taskDefinition = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The task definition name. If not provided will attempt "
                     "to get from environment variable ONLINE_TRAINING_TASK. "
                     "See boto3 docs for run_task arg `taskDefinition`"))
    container = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The container name. If not provided will attempt "
                     "to get from environment variable "
                     "ONLINE_TRAINING_CONTAINER. See boto3 docs for "
                     "run_task arg `taskDefinition`"))
    subnet = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The subnet. If not provided, will attempt to get from "
                     "environment variable ONLINE_TRAINING_SUBNET. See boto3 "
                     "docs for run_task arg `subnets` under "
                     "`network_configuration`)"))
    securityGroup = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The security group. If not provided, will attempt to "
                     "get from environment variable ONLINE_TRAINING_SECURITY. "
                     "See boto3 docs for run_task arg `securityGroups` under "
                     "`network_configuration`)"))
    trackingURI = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("mlflow tracking URI. If not provided, will attempt to "
                     "get environment variable MLFLOW_TRACKING_URI."))
    training_args = argschema.fields.Str(
        required=True,
        description=("(as string) dict with keys and values to pass to the "
                     "container-hosted croissant training module. Will be "
                     "parsed inside the container against "
                     "`croissant.train.TrainingSchema`. Proper CLI quote "
                     "formatting: --training_args '{\"arg1\": \"val1\", }'"))
    experiment_name = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("name of mlflow experiment. If not provided, will "
                     "attempt to get environment variable MLFLOW_EXPERIMENT. "
                     "It is assumed that this experiment was previously "
                     "created with an s3 artifact URI. If not, a new "
                     "experiment will be created, but the artifact will be "
                     "local to the container and lost at job completion."))

    @mm.post_load
    def env_populate(self, data, **kwargs):
        lookup = {
                "cluster": "ONLINE_TRAINING_CLUSTER",
                "taskDefinition": "ONLINE_TRAINING_TASK",
                "container": "ONLINE_TRAINING_CONTAINER",
                "subnet": "ONLINE_TRAINING_SUBNET",
                "securityGroup": "ONLINE_TRAINING_SECURITY",
                "trackingURI": "MLFLOW_TRACKING_URI",
                "experiment_name": "MLFLOW_EXPERIMENT"}
        for k, v in lookup.items():
            if data[k] is None:
                if v not in os.environ.keys():
                    raise OnlineTrainingException(
                            f"{k} was not specified and {v} is not an "
                            "ENV variable")
                data[k] = os.environ.get(v)
        return data


class OnlineTrainingOutputSchema(argschema.schemas.DefaultSchema):
    response = argschema.fields.Dict(
        required=True,
        description="boto3 response to run_task call")


class OnlineTraining(argschema.ArgSchemaParser):
    default_schema = OnlineTrainingSchema
    default_output_schema = OnlineTrainingOutputSchema

    def run(self):
        client = boto3.client('ecs')
        # we don't have databricks or kubernetes set up
        # mlflow will always run with local backend
        command = ["--backend", "local", "--experiment-name",
                   self.args['experiment_name']]
        # format the training args so mlflow passes them through to the module
        for k, v in json.loads(self.args['training_args']).items():
            command.append("-P")
            command.append(f"{k}={v}")

        response = client.run_task(
            cluster=self.args['cluster'],
            launchType="FARGATE",
            taskDefinition=self.args['taskDefinition'],
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": [self.args['subnet']],
                    "securityGroups": [self.args['securityGroup']],
                    "assignPublicIp": "ENABLED"}},
            overrides={
                "containerOverrides": [
                    {
                        "name": self.args['container'],
                        "command": command,
                        "environment": [
                            {
                                "name": "MLFLOW_TRACKING_URI",
                                "value": self.args['trackingURI']}]
                            }
                    ]
                }
            )

        # the response has some datetime objects, we can force those to str
        self.output({'response': response}, default=str, indent=2)


if __name__ == "__main__":  # pragma nocover
    online = OnlineTraining()
    online.run()
