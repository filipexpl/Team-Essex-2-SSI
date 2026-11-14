FROM mcr.microsoft.com/devcontainers/python:3.11
# liblsl for pylsl (live streaming); no Bluetooth here - hardware runs on the host laptop.
RUN apt-get update && apt-get install -y --no-install-recommends libpugixml1v5 \
    && rm -rf /var/lib/apt/lists/*
