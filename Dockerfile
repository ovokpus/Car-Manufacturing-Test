# Build a self-contained prediction service.
#
# Training runs during the image build, on the train.csv that ships with this
# repo, so `docker compose up` serves predictions without a separate train step.
# The fit takes about a second; the slow part of the build is installing wheels.
#
# The dataset is the Kaggle "Mercedes-Benz Greener Manufacturing" train file.
# See README.md if a checkout does not contain it.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY greener ./greener
RUN pip install --no-cache-dir --no-deps .

COPY train.csv ./train.csv
RUN python -m greener.train --train /app/train.csv --output /app/artifacts/model.joblib

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "greener.api:app", "--host", "0.0.0.0", "--port", "8000"]
