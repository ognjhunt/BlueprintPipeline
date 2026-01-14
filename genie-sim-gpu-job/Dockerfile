# Genie Sim GPU Execution Job Dockerfile
# Builds on NVIDIA Isaac Sim with Genie Sim repo installed for GPU-backed local runs.

FROM nvcr.io/nvidia/isaac-sim:2024.1.0

ENV ISAAC_SIM_PATH=/isaac-sim
ENV GENIESIM_ROOT=/opt/geniesim

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG GENIESIM_REPO=https://github.com/AgibotTech/genie_sim.git
ARG GENIESIM_REF=main

RUN git clone --depth 1 --branch ${GENIESIM_REF} ${GENIESIM_REPO} ${GENIESIM_ROOT}

WORKDIR /app

COPY genie-sim-local-job/requirements.txt /app/requirements.txt
COPY genie-sim-local-job/requirements.lock /app/requirements.lock

RUN ${ISAAC_SIM_PATH}/python.sh -m pip install --upgrade pip \
    && ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir -r /app/requirements.txt -c /app/requirements.lock \
    && ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir -r ${GENIESIM_ROOT}/requirements.txt

COPY genie-sim-submit-job/ /app/genie-sim-submit-job/
COPY genie-sim-export-job/ /app/genie-sim-export-job/
COPY tools/ /app/tools/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMNI_KIT_ALLOW_ROOT=1

ENTRYPOINT ["/isaac-sim/python.sh", "/app/genie-sim-submit-job/submit_to_geniesim.py"]
