FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY proxy.py ./

RUN pip install --no-cache-dir .

ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
EXPOSE 7187

CMD ["python", "proxy.py"]
