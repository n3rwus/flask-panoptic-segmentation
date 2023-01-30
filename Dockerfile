# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

WORKDIR /engineering-project
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000

COPY . .
CMD ["python3", "-m", "flask", "run"]