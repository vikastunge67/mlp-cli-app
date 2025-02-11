### Dockerfile ###
FROM python:3.12

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Set the default command to run CLI
ENTRYPOINT ["python", "cli.py"]
 