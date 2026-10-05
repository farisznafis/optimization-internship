# Batch image for the optimization CLI.
#   docker build -t optimization-internship .
#   docker run --rm optimization-internship course-selection
#   docker run --rm --env-file .env -v "$PWD/outputs:/app/outputs" optimization-internship task-assignment
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY projects ./projects

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/outputs \
    && chown app:app /app/outputs
USER app

ENTRYPOINT ["optim"]
CMD ["--help"]
