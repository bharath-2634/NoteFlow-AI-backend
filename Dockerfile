# Use a specific, stable Python image as the base
# 'slim-buster' is a smaller image, good for deployment
FROM python:3.10-slim-buster

# Set environment variables for non-sensitive data within the container
# PYTHONUNBUFFERED=1 ensures Python output is not buffered, good for logs
# PORT=8000 is the default port Uvicorn will listen on
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies for document parsing (if needed by textract for .doc/.ppt)
# UNCOMMENT AND ADD THESE IF YOU ENABLED .doc/.ppt SUPPORT AND textract IS USED
# If you don't use textract for .doc/.ppt, you can comment or remove these lines.
# RUN apt-get update -y && apt-get install -y antiword unoconv libreoffice

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker layer caching
# If requirements.txt doesn't change, this layer is cached, speeding up builds
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that the application will listen on
EXPOSE ${PORT}

# Command to run the application when the container starts
# --host 0.0.0.0 is crucial for Docker containers to listen on all interfaces
# --port $PORT uses the environment variable defined above
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]