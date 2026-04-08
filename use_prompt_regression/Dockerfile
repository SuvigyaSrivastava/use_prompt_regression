FROM python:3.11-slim

WORKDIR /app

COPY use_prompt_regression/server/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir \
    openenv-core \
    openai \
    pydantic \
    fastapi \
    uvicorn \
    python-dotenv

COPY use_prompt_regression/ ./use_prompt_regression/

ENV PYTHONPATH=/app/use_prompt_regression/server
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

CMD ["uvicorn", "use_prompt_regression.server.app:app", "--host", "0.0.0.0", "--port", "8000"]