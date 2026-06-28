# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies including PostgreSQL client and nss-wrapper for rootless environments
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    libnss-wrapper \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory inside the container
WORKDIR /code

# Copy and install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Create required directories and mock user/group configuration files for rootless environments
RUN mkdir -p /data/postgres /code/sockets && \
    touch /code/passwd /code/group && \
    chmod 666 /code/passwd /code/group && \
    chown -R 1000:1000 /code /data

# Switch to non-root user (UID 1000) for security compliance on platforms like Hugging Face Spaces
USER 1000

# Expose the internal port (informational metadata)
EXPOSE 7860

# Run the FastAPI application using Uvicorn production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
