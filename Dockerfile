FROM python:3.11
LABEL description="project name"
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt