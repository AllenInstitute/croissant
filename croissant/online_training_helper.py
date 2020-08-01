import argschema
import boto3
from botocore.exceptions import NoCredentialsError
import json
import logging
import mlflow
import multiprocessing


class EnvHelperException(Exception):
    pass


class EnvHelperSchema(argschema.ArgSchema):
    stack = argschema.fields.Str(
        required=True,
        description="stack name from which to suggest environment variables")
    timeout = argschema.fields.Float(
        required=False,
        default=20.0,
        description=("timeout for mlflow tracking client list_experiments "
                     "call. Without a VPN connection, this call hangs."))


def mlflow_experiment_check(tracking_uri):
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiments = client.list_experiments()
    exoptions = ""
    for e in experiments:
        if e.artifact_location.startswith("s3://"):
            exoptions += (f"\nexport MLFLOW_EXPERIMENT={e.name}")
    return exoptions


class EnvHelper(argschema.ArgSchemaParser):
    default_schema = EnvHelperSchema

    def run(self):
        client = boto3.client('cloudformation')
        try:
            desc = client.describe_stacks(StackName=self.args['stack'])
        except NoCredentialsError as e:
            logging.error(str(e))
            logging.error("no AWS credentials found.")
            return

        if len(desc['Stacks']) != 1:
            raise EnvHelperException(f"description for {self.args['stack']} "
                                     "did not return exactly one entry")
        outs = {i['OutputKey']: i['OutputValue']
                for i in desc['Stacks'][0]['Outputs']}

        exports = ""
        lookup = {
                'ONLINE_TRAINING_CLUSTER': outs['FargateClusterName'],
                'ONLINE_TRAINING_TASK': outs['TaskDefinitionName'],
                'ONLINE_TRAINING_CONTAINER': outs['ContainerNameOutput'],
                'ONLINE_TRAINING_SUBNET': outs['PrivateSubnet'],
                'ONLINE_TRAINING_SECURITY': outs['VPCUserSecurityGroupId']}
        sets = [f"{k}={v}" for k, v in lookup.items()]
        exports = [f"export {s}" for s in sets]

        client = boto3.client('secretsmanager')
        secret = json.loads(
                client.get_secret_value(
                    SecretId=outs['DBSecret'])['SecretString'])
        tracking = (f"{secret['engine']}://{secret['username']}:"
                    f"{secret['password']}@{secret['host']}:{secret['port']}/"
                    f"{secret['dbname']}")
        tracking = tracking.replace("postgres", "postgresql")
        exports.append("export MLFLOW_TRACKING_URI=" + tracking)
        exports = "\n".join(exports)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(f"ENV variables to use stack {self.args['stack']}"
                    f"\n\n{exports}\n")

        # this call hangs without a VPN connection, so give it a timeout
        pool = multiprocessing.Pool(processes=1)
        result = pool.apply_async(mlflow_experiment_check, (tracking,))
        try:
            exoptions = result.get(timeout=self.args['timeout'])
        except multiprocessing.context.TimeoutError:
            logging.warning("check for mlflow experiments timed out after "
                            f"{self.args['timeout']} seconds. Perhaps you are "
                            "not connected to the client VPN.")
        else:
            logger.info("Available experiments with s3 artifact stores:"
                        f"\n{exoptions}")


if __name__ == "__main__":
    eh = EnvHelper()
    eh.run()
