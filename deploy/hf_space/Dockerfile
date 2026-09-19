# SatQuery API as a Hugging Face Docker Space (free CPU tier: 2 vCPU, 16 GB).
#
# Runs the `cpu` profile (configs/profiles/cpu.yaml) with the CPU deployment
# map (configs/deploy.cpu.yaml): every specialist model, the 3B VLM shed.
# The Space repo holds only this Dockerfile, start.sh and README.md; the
# code is cloned from GitHub at build time and the weights (≈ 1.4 GB) are
# downloaded from a private Hugging Face model repo at container start,
# because a free Space has no persistent disk and the weights must not live
# in a public repo (DIOR-RSVG-derived grounding weights are CC-BY-NC).
#
# Space settings:
#   Secrets   HF_TOKEN                 read token for the weights repo
#   Variables SATQUERY_WEIGHTS_REPO    e.g. <user>/satquery-cpu-weights
#             SATQUERY_CORS_ORIGINS    https://satquery-ai.vercel.app
#             SATQUERY_GIT_REF         main (or a tag / commit)

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgl1 git curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000; /app must be writable by it.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

ARG SATQUERY_GIT_REF=main
RUN git clone --depth 1 --branch ${SATQUERY_GIT_REF} https://github.com/aditya-coder07/SatQuery_AI.git /app \
    && rm -rf /app/.git /app/frontend /app/notebooks

RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt \
    && pip install --no-cache-dir --timeout 120 --retries 5 torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir timm==1.0.29 "huggingface_hub>=0.30"

COPY start.sh /app/deploy/hf_space/start.sh
RUN chmod +x /app/deploy/hf_space/start.sh && mkdir -p /app/checkpoints /app/artifacts && chown -R user:user /app

USER user
ENV HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=1 \
    SATQUERY_PROFILE=cpu \
    SATQUERY_CORS_ORIGINS=https://satquery-ai.vercel.app \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["/app/deploy/hf_space/start.sh"]
