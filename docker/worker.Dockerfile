# Python 3.12, not 3.11: rasterio 1.5.1 publishes no wheels below 3.12, so a
# 3.11 base cannot resolve the pinned requirement set and the build fails at
# pip install. CI pins 3.12 for the same reason; the images had drifted from it.
FROM python:3.12-slim

# rasterio and pillow wheels are manylinux, but GDAL's runtime bits and the
# image codecs are not all vendored; without these the import succeeds and the
# first raster read fails, which is a much worse place to discover it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . .

CMD ["python", "-m", "satquery.controller.worker"]
