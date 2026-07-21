FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_COMPILE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-compile --index-url https://download.pytorch.org/whl/cpu "torch==2.12.1+cpu"
RUN sed -n '/^pandas/Ip;/^numpy/Ip;/^scikit-learn/Ip;/^networkx/Ip;/^pyyaml/Ip;/^matplotlib/Ip;/^seaborn/Ip;/^scipy/Ip;/^pytest/Ip' requirements.txt > /tmp/requirements-core.txt \
    && pip install --no-compile -r /tmp/requirements-core.txt
RUN sed -n '/^torch-geometric/Ip' requirements.txt > /tmp/requirements-pyg.txt \
    && pip install --no-compile -r /tmp/requirements-pyg.txt
RUN sed -n '/^chromadb/Ip;/^sentence-transformers/Ip;/^mcp/Ip;/^pydantic/Ip' requirements.txt > /tmp/requirements-services.txt \
    && pip install --no-compile -r /tmp/requirements-services.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    MPLCONFIGDIR=/cache/matplotlib \
    TRANSFORMERS_VERBOSITY=error \
    TQDM_DISABLE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . .
RUN mkdir -p train models saved_results /cache/huggingface /cache/torch /cache/matplotlib \
    && python -c "import matplotlib.font_manager"

CMD ["python", "scripts/run_inference.py"]
