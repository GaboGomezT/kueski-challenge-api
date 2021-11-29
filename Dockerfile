FROM python:3.9.8-buster

COPY ./app /src
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN apt update
RUN apt install curl -y
RUN apt install unzip -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

WORKDIR /src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

