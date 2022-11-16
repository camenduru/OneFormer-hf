FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        git \
        make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev  \
    	ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx \
		&& rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash admin
USER admin

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# ENV HOME=/home/user \
# 	PATH=/home/user/.local/bin:$PATH

# RUN curl https://pyenv.run | bash
# ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
# RUN pyenv install 3.8.15 && \
#     pyenv global 3.8.15 && \
#     pyenv rehash && \
#     pip install --no-cache-dir --upgrade pip setuptools wheel

ENV WORKDIR=/code
WORKDIR $WORKDIR
RUN chown -R admin:admin $WORKDIR
RUN chmod 755 $WORKDIR

# RUN nvidia-smi


COPY requirements.txt $WORKDIR/requirements.txt
COPY oneformer $WORKDIR/oneformer

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

RUN pwd
RUN ls

RUN sh deform_setup.sh

USER admin

EXPOSE 7860

ENTRYPOINT ["python", "gradio_app.py"]
