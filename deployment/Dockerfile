# Base image
FROM python:3.10.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    ffmpeg \
    portaudio19-dev \
    && apt-get clean

# Create a user with UID 1000
RUN useradd -m -u 1000 user

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip==23.1.2

# Install Cython first (if needed)
RUN pip install Cython

# Install the dependencies from requirements.txt excluding flask-restplus
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of your application code
COPY . /app

# Command to run the application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
