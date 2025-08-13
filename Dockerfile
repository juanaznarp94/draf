FROM hashicorp/terraform:light


RUN apk add --no-cache \
    openssh \
    curl \
    bash \
    python3 \
    py3-pip \
    unzip \
    jq


WORKDIR /workspace


COPY deploy.sh .
COPY destroy.sh .
COPY terraform ./terraform
COPY code ./code
COPY Flower ./Flower
RUN find . -type f -exec dos2unix {} +

RUN mkdir -p /root/.ssh

RUN mkdir -p volume
RUN mkdir -p volume/terraform
RUN chmod +x deploy.sh destroy.sh


ENTRYPOINT ["/bin/bash"]
