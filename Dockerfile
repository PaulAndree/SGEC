FROM python:3.11.4

WORKDIR /usr/src/app

COPY ./modelfile ./modelfile
COPY ./prompt_templates ./prompt_templates
COPY ./utils ./utils
COPY .env .
COPY main_API.py .
COPY ollama_client.py .
COPY Requirements.txt .

RUN pip install -r Requirements.txt
EXPOSE 8000

EXPOSE 8000

CMD ["uvicorn", "main_API:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "500", "--reload"]

#docker build -t "backend" .
#docker images
#docker run -ti