FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud: Override DATA_PATH and OUTPUT_PATH with env vars
# e.g. -e DATA_PATH=s3://my-bucket/data
ENV DATA_PATH=/app/data
ENV OUTPUT_PATH=/app/outputs
ENV MLFLOW_URI=/app/mlruns

EXPOSE 8501

# Default: run Streamlit dashboard
CMD ["streamlit", "run", "src/serving/dashboard.py",
     "--server.port=8501", "--server.address=0.0.0.0"]
