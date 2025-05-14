FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages with specific versions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.21.0 \
    pymc3==3.11.5 \
    theano-pymc==1.1.2 \
    hddm==0.9.8

# Set working directory
WORKDIR /app

# Copy your code into the container
COPY . .

# Command to run when the container starts
CMD ["bash"]
