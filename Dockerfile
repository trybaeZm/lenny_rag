FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Start the app using the PORT environment variable
CMD ["python", "run.py"]