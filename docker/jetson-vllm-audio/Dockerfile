ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.22.0-ubuntu2404
FROM ${VLLM_BASE_IMAGE}

COPY requirements.txt /tmp/ragent-audio-requirements.txt
RUN python3 -m pip install --no-cache-dir \
      -r /tmp/ragent-audio-requirements.txt \
    && rm /tmp/ragent-audio-requirements.txt
