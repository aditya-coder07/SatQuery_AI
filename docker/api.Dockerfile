# Python 3.12, not 3.11: rasterio 1.5.1 publishes no wheels below 3.12, so a
# 3.11 base cannot resolve the pinned requirement set and the build fails at
# pip install. CI pins 3.12 for the same reason; the images had drifted from it.
FROM python:3.12-slim AS cpu-image

# rasterio and pillow wheels are manylinux, but GDAL's runtime bits and the
# image codecs are not all vendored; without these the import succeeds and the
# first raster read fails, which is a much worse place to discover it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements if present, otherwise just continue
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . .

# Serve as a non-root user. The application writes two things inside the
# image - `artifacts/<run_id>` next to the working directory and its uploads
# under the system temp directory - so /app is chowned rather than left
# root-owned, which would turn the first query into a permission error.
RUN useradd --create-home --uid 1000 satquery \
    && mkdir -p /app/artifacts \
    && chown -R satquery:satquery /app
USER satquery

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "satquery.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Optional GPU image stage
# Inherits the fully built CPU image and injects the heavy inference stack.
# Reuses all layers from cpu-image without duplicating the Dockerfile.
FROM cpu-image AS gpu-image

USER root
# Install the exact training/inference dependencies verified in the host environment.
# Using lower-bound logic from pyproject.toml but locking exactly to the proven versions.
# Install gcc and g++ for triton JIT compilation
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    torch==2.13.0+cu126 \
    torchvision==0.28.0+cu126 \
    transformers==5.15.1 \
    peft==0.20.0 \
    bitsandbytes==0.50.2 \
    accelerate==1.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cu126

USER satquery
