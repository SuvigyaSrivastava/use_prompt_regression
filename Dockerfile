FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    openenv-core \
    openai \
    pydantic \
    fastapi \
    uvicorn \
    python-dotenv

COPY server/ ./server/

ENV PYTHONPATH=/app/server
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]