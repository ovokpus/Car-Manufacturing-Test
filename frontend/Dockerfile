# Browser UI. It only talks to the API; the model stays in the api image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_URL=http://api:8000 \
    EXAMPLES_DIR=/app/examples \
    BANNER_PATH=/app/assets/banner.jpg

WORKDIR /app

COPY frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY frontend /app/frontend
COPY examples /app/examples
COPY assets/banner.jpg /app/assets/banner.jpg
COPY .streamlit /app/.streamlit

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
