FROM continuumio/miniconda3

ARG MYBRANCH

RUN git clone https://github.com/AllenInstitute/croissant

# establish requirements at container build time
# install croissant for CI/CD testing of container
RUN cd croissant && \
    git checkout ${MYBRANCH} && \
    conda env create -f conda.yaml -n myenv && \
    conda run -n myenv pip install .[ALL,TEST]

RUN conda clean -ayv && \
    find /opt/conda -follow -type f -regextype posix-extended -regex '.*\.(pyc)' -delete

WORKDIR /croissant
ENTRYPOINT ["conda", "run", "-n", "myenv", "mlflow", "run", "--no-conda", "/croissant"]
