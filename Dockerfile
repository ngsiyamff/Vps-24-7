FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git curl wget tmux nano ttyd && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
CMD ["bash", "start.sh"]
