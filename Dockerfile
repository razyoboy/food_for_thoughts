#!/usr/bin/env python
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

EXPOSE $PORT

RUN pip install --no-cache-dir -r requirements.txt
CMD python3 -u "src/main.py"
