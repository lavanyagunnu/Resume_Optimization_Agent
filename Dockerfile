# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Make ports accessible
EXPOSE 9999

# Install Python dependencies if needed
# RUN pip install -r requirements.txt

# Default command to run server
CMD ["python", "server.py"]
