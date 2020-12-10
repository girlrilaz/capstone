FROM python:3.8

WORKDIR /code

COPY ./data/ /code/data/
COPY ./helper/ /code/helper/
COPY ./logs/ /code/logs/
COPY ./models/ /code/models/
COPY ./reports/ /code/reports/
COPY ./tests/ /code/tests/
COPY app.py /code/
COPY requirements.txt /code/

RUN pip install -r requirements.txt

CMD [ "python", "./app.py" ]


