FROM lalia-moss-cuda-probe:20261027-r4
COPY requirements.lock verify_environment.py /opt/frontend/
RUN python /opt/frontend/verify_environment.py save \
    && python -m pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r /opt/frontend/requirements.lock \
    && python -m pip check \
    && python /opt/frontend/verify_environment.py verify \
    && python -c 'import librosa, soundfile, soxr, numba; from transformers.audio_utils import load_audio; from transformers.models.whisper.modeling_whisper import WhisperEncoder'
