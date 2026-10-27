FROM python@sha256:04b6d37c0976bcb0757d6bf6819ee4b7a4094497717896bc002f845b9c06c752
RUN apt-get update && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -r /var/lib/apt/lists/*
RUN python -m venv /opt/probe
ENV PATH=/opt/probe/bin:$PATH
COPY repair_cusparselt.py /opt/repair_cusparselt.py
RUN python -m pip install --no-cache-dir \
    'https://files.pythonhosted.org/packages/2e/29/69cfbb602cd91690c55d38ba9fe53e6a7e76a6fa647bf38f19c138d25449/wheel-0.48.0-py3-none-any.whl#sha256=3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab' \
    'https://files.pythonhosted.org/packages/b7/b9/c538f279a4e237a006a2c98387d081e9eb060d203d8ed34467cc0f0b9b53/packaging-26.0-py3-none-any.whl#sha256=b36f1fef9334a5588b4166f8bcd26a14e521f2b55e6b9de3aaa80d3ff7a37529' \
    && python -m pip download --no-cache-dir --no-deps --dest /tmp/original \
    'https://files.pythonhosted.org/packages/46/e1/cdc1797eadf82d3a9a575a19b33fdc871a97edbec42c00b5b5e914f4aff4/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl#sha256=4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f' \
    && python /opt/repair_cusparselt.py /tmp/original/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl /tmp/repaired \
    && python -m pip install --no-cache-dir --no-deps /tmp/repaired/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl \
    && cp /tmp/repaired/repair-result.json /opt/cusparselt-repair.json \
    && rm -r /tmp/original /tmp/repaired
RUN python -m pip install --no-cache-dir \
    'https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6afc5e5944fb0bd927e08d32a11f71d19154001f1c81fa00a6ed81a217451795' \
    && python -m pip install --no-cache-dir 'transformers==5.17.0' 'safetensors==0.8.0' 'numpy==2.3.5' \
    && python -m pip check
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "/probe/score_decoder.py"]

RUN apt-get update && apt-get install --yes --no-install-recommends gcc libc6-dev \
    && rm -r /var/lib/apt/lists/* \
    && python -c 'import pathlib, sysconfig; assert (pathlib.Path(sysconfig.get_path("include")) / "Python.h").is_file()' \
    && gcc --version \
    && dpkg-query -W gcc libc6-dev \
    && python -m pip check
