FROM python:3.11-slim


WORKDIR /Flower


COPY __init__.py .
COPY federated_ex ./federated_ex
COPY initToml.py .
COPY pyproject.toml .

COPY entrypoint.sh .
COPY requirements.txt .
COPY .env .
RUN apt update
RUN apt install -y git
RUN pip install git+https://github.com/gngdb/pytorch-minimize.git
RUN pip install python-dotenv
RUN python initToml.py
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .

EXPOSE 9093

ENTRYPOINT ["/bin/bash"]



