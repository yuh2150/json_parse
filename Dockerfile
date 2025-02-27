FROM python:3.10-slim

# Install build dependencies for building C++ extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 6006

CMD ["python", "/app/booking_details.py"]
