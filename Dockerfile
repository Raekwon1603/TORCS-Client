# Default base image
ARG BASE_IMAGE="continuumio/miniconda3:24.4.0-0"
FROM $BASE_IMAGE

ENV LANG=C.UTF-8
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

RUN conda config --add channels conda-forge

RUN conda install -y python=3.10.7 && \
    conda install -y conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda update -c defaults conda -y && \
    conda install -y -c pytorch cpuonly pytorch=2.3.0 && \
    conda install -y -c conda-forge pandas=2.2.2 scikit-learn=1.2.2

USER 1001

ARG DIR_NAME="torcs-agent"

ARG MODEL_PATH="omni_model/model_epoch_400.pt"
ARG SCALAR_PATH="omni_model/scalar_epoch_400.pkl"

ADD NeuralNet.py /${DIR_NAME}/NeuralNet.py
ADD NeuralNetSettings.py /${DIR_NAME}/NeuralNetSettings.py

ADD ./models/${MODEL_PATH} /${DIR_NAME}/models/${MODEL_PATH}
ADD ./models/${SCALAR_PATH} /${DIR_NAME}/models/${SCALAR_PATH}

COPY ./driver /${DIR_NAME}/driver

WORKDIR /${DIR_NAME}/driver

CMD [ "localhost" ]
ENTRYPOINT ["python", "run.py"]