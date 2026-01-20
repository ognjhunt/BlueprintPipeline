# Genie Sim GPU Execution Job Dockerfile
# Builds on NVIDIA Isaac Sim with Genie Sim repo installed for GPU-backed local runs.

ARG ISAAC_SIM_VERSION=2024.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

ENV ISAAC_SIM_PATH=/isaac-sim
ENV GENIESIM_ROOT=/opt/geniesim

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG GENIESIM_REPO=https://github.com/AgibotTech/genie_sim.git
ARG GENIESIM_REF

COPY GENIESIM_REF /tmp/GENIESIM_REF

RUN GENIESIM_REF=${GENIESIM_REF:-$(cat /tmp/GENIESIM_REF)} \
    && git clone --no-checkout ${GENIESIM_REPO} ${GENIESIM_ROOT} \
    && cd ${GENIESIM_ROOT} \
    && git fetch --depth 1 origin ${GENIESIM_REF} \
    && git checkout ${GENIESIM_REF}

WORKDIR /app

COPY genie-sim-local-job/requirements.txt /app/requirements.txt
COPY genie-sim-local-job/requirements.lock /app/requirements.lock
COPY genie-sim-gpu-job/requirements.geniesim.lock /app/requirements.geniesim.lock

RUN ${ISAAC_SIM_PATH}/python.sh -m pip install --upgrade pip \
    && ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir -r /app/requirements.txt -c /app/requirements.lock \
    && ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir -r ${GENIESIM_ROOT}/requirements.txt -c /app/requirements.geniesim.lock

COPY genie-sim-submit-job/ /app/genie-sim-submit-job/
COPY genie-sim-export-job/ /app/genie-sim-export-job/
COPY genie-sim-gpu-job/entrypoint.sh /app/genie-sim-gpu-job/entrypoint.sh
COPY genie-sim-gpu-job/gpu_warmup.py /app/genie-sim-gpu-job/gpu_warmup.py
COPY tools/ /app/tools/

RUN chmod +x /app/genie-sim-gpu-job/entrypoint.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMNI_KIT_ALLOW_ROOT=1

ENTRYPOINT ["/app/genie-sim-gpu-job/entrypoint.sh"]
