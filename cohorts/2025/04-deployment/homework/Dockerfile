# Use an official Python base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY starter.py .

# Set the entrypoint (optional, for CLI usage)
ENTRYPOINT ["python", "starter.py"]