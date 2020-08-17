import boto3
import os
import argschema
import marshmallow as mm
from croissant.train import TrainingSchema


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
                     "docs for run_task arg `cluster` "
                     "`https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecs.html#ECS.Client.run_task`"))  # noqa
    taskDefinition = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The task definition name. If not provided will attempt "
                     "to get from environment variable ONLINE_TRAINING_TASK. "
                     "See boto3 docs for run_task arg `taskDefinition` "
                     "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecs.html#ECS.Client.run_task"))  # noqa
    container = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The container name. If not provided will attempt "
                     "to get from environment variable "
                     "ONLINE_TRAINING_CONTAINER. See boto3 docs for "
                     "run_task arg `overrides` "
                     "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecs.html#ECS.Client.run_task"))  # noqa
    subnet = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The subnet. If not provided, will attempt to get from "
                     "environment variable ONLINE_TRAINING_SUBNET. See boto3 "
                     "docs for run_task arg `subnets` under "
                     "`network_configuration` "
                     "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecs.html#ECS.Client.run_task"))  # noqa
    securityGroup = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("The security group. If not provided, will attempt to "
                     "get from environment variable ONLINE_TRAINING_SECURITY. "
                     "See boto3 docs for run_task arg `securityGroups` under "
                     "`network_configuration` "
                     "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecs.html#ECS.Client.run_task"))  # noqa
    trackingURI = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=True,
        description=("mlflow tracking URI. If not provided, will attempt to "
                     "get environment variable MLFLOW_TRACKING_URI. See:"
                     "https://mlflow.org/docs/latest/cli.html#cmdoption-mlflow-run-b"))  # noqa
    container_image = argschema.fields.Str(
        required=False,
        description=("if provided, will define a task to run using this str "
                     "as the image specification for the task definition. "
                     "An example format of this str is: "
                     "'docker.io/alleninstitutepika/croissant:<tag>'. See: "
                     "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html#container_definition_image"))  # noqa
    training_args = argschema.fields.Nested(
        TrainingSchema,
        required=True,
        description="nested croissant.train.TrainingSchema")
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
        for k, v in self.args['training_args'].items():
            # we're passing explicit args to the container
            if k not in ['input_json', 'output_json']:
                command.append("-P")
                command.append(f"{k}={v}")

        task_def_arn = self.args['taskDefinition']
        container_name = self.args['container']

        # dynamically specifying a different container requires
        # defining a new ECS task
        if "container_image" in self.args:
            # duplicate key parameters from stack-defined task
            task = client.describe_task_definition(
                    taskDefinition=self.args['taskDefinition'])
            taskdef = task['taskDefinition']
            cdefs = taskdef['containerDefinitions']
            assert len(cdefs) == 1

            # set the image and a new name
            cdefs[0]['image'] = self.args['container_image']
            cdefs[0]['name'] = 'dynamic-container'
            response = client.register_task_definition(
                    family="dynamic-image",
                    taskRoleArn=taskdef['taskRoleArn'],
                    executionRoleArn=taskdef['executionRoleArn'],
                    containerDefinitions=cdefs,
                    cpu=taskdef['cpu'],
                    memory=taskdef['memory'],
                    requiresCompatibilities=['FARGATE'],
                    networkMode='awsvpc')

            task_def_arn = response['taskDefinition']['taskDefinitionArn']
            container_name = \
                response['taskDefinition']['containerDefinitions'][0]['name']

        response = client.run_task(
            cluster=self.args['cluster'],
            launchType="FARGATE",
            taskDefinition=task_def_arn,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": [self.args['subnet']],
                    "securityGroups": [self.args['securityGroup']],
                    "assignPublicIp": "ENABLED"}},
            overrides={
                "containerOverrides": [
                    {
                        "name": container_name,
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
