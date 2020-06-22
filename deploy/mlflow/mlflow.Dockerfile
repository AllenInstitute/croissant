# Docker container to simplify running MLFlow UI, connected to cloud resources
FROM python:3.8.0

RUN pip install \
    mlflow>=1.8.0 \
    psycopg2-binary

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

RUN apt-get update && apt-get -y install jq

# Get secret info and connect to database
# Must be using VPN, in VPC, or have public endpoint for backend store
CMD secret=$(aws secretsmanager get-secret-value --secret-id test/mlflow --query SecretString --output text) &&\
    USER=$(echo $secret | jq .username -r) && \
    PASSWORD=$(echo $secret | jq .password -r) && \
    HOST=$(echo $secret | jq .host -r) && \
    PORT=$(echo $secret | jq .port -r) && \
    mlflow ui \
      --host 0.0.0.0 \
      --backend-store-uri postgresql://$USER:$PASSWORD@$HOST:$PORT
