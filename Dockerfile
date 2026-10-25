# Dashboard image. Build after the data has been built locally (data/processed must exist):
#   uv run python -m cifresignificative.fetch && uv run python -m cifresignificative.clean.build
#   docker build -t italy-vs-eu27 . && docker run -p 8501:8501 italy-vs-eu27
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY .streamlit ./.streamlit
COPY docs ./docs
COPY reports/findings.md ./reports/findings.md
COPY data/processed ./data/processed
EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/cifresignificative/dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
