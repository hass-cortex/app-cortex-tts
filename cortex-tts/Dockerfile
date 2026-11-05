# The uv binary is copied in from its own published image; naming the
# stage is what lets `COPY --from` reference it.
FROM ghcr.io/astral-sh/uv:0.9.9 AS uv

# A literal FROM: Supervisor injects an Alpine BUILD_FROM default an ARG would accept.
FROM ghcr.io/hassio-addons/debian-base:9.3.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:${PATH}"

# libgomp1 is ONNX Runtime's OpenMP runtime; libsndfile1 backs soundfile.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gosu \
        libgomp1 \
        libsndfile1 \
        libstdc++6 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false cortex-tts

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first: the layer only rebuilds when the lockfile moves, not on
# every source edit. One extra carries the model that is not pure ONNX
# Runtime: `omnivoice` adds torch, torchaudio, transformers, librosa and pydub
# for the half of OmniVoice the ONNX graph does not replace, and is most of
# the image. The two ONNX-only models — the 40M and MOSS — touch none of it.
COPY pyproject.toml uv.lock README.md ./
RUN uv venv /opt/venv \
    && uv sync --frozen --no-dev --extra omnivoice --active --no-install-project \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /root/.cache

COPY src/ /app/src/
COPY web/ /app/web/
RUN chown -R cortex-tts:cortex-tts /app

COPY rootfs /

# The Supervisor restarts a container whose health check fails; the app's
# own liveness probe is the check. A long start period covers the first
# download of the default model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8771/health || exit 1

ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_REF
ARG BUILD_VERSION

LABEL \
    io.hass.name="Cortex TTS" \
    io.hass.description="On-device text-to-speech for Home Assistant — a choice of models with built-in, designed and cloned voices, and the Chinese text pipeline they lack" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="hass-cortex" \
    org.opencontainers.image.title="Cortex TTS" \
    org.opencontainers.image.description="On-device text-to-speech for Home Assistant — a choice of models with built-in, designed and cloned voices, and the Chinese text pipeline they lack" \
    org.opencontainers.image.vendor="hass-cortex" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/hass-cortex/app-cortex-tts" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${BUILD_REF}" \
    org.opencontainers.image.version="${BUILD_VERSION}"
