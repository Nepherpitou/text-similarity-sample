FROM python:3.9-slim

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN python -m nltk.downloader punkt

COPY ./src /code/src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
