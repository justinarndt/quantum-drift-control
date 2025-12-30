FROM python:3.10-slim

# Install system dependencies for Stim/Bazel
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Bazelisk (Bazel launcher)
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

# Python Deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy Source
WORKDIR /app
COPY . .

# Default entry point
CMD ["bazel", "test", "//tests:all"]