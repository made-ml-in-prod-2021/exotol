ARG PYTHON_VERSION=3.8

#FROM python:${PYTHON_VERSION}-slim AS compile-image
FROM python:slim-buster AS compile-image

ENV PIP_NO_CACHE_DIR=off
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./requirements.txt

RUN  apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && pip install -r requirements.txt


#FROM python:${PYTHON_VERSION}-slim AS build-image
FROM python:slim-buster AS build-image

COPY --from=compile-image /opt/venv /opt/venv
WORKDIR .
COPY models/model_rf.pkl ./model.pkl
COPY ./heart_disease_app/ ./heart_disease_app/
ENV PATH="opt/venv/bin:$PATH"
ENV PATH_TO_MODEL="/model.pkl"
ENV PORT=8000
ENV PATH_TO_LOGGING_CONF="heart_disease_app/logging.yaml"

CMD uvicorn heart_disease_app.app:app --host 0.0.0.0 --port ${PORT}