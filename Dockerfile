FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python venv module (usually included in slim, but just in case)
RUN apt-get update && apt-get install -y python3-venv && rm -rf /var/lib/apt/lists/*

# Create a virtual environment inside the container
RUN python -m venv /opt/venv

# Make sure the venv's python and pip are used
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies into the venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Start the app using the PORT environment variable
CMD ["python", "run.py"]
