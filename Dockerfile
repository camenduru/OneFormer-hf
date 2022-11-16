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

# RUN apt-get -y update
# RUN apt install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get -y install python3.8
# RUN apt-get -y install python3-pip
# RUN apt install -y python3.8-distutils
# RUN apt-get install -y gcc
# RUN apt-get install -y python3.8-dev

# RUN useradd -ms /bin/bash admin
# USER admin

RUN useradd -ms /bin/bash user
USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
RUN pyenv install 3.8.15 && \
    pyenv global 3.8.15 && \
    pyenv rehash && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

ENV WORKDIR=/code
WORKDIR $WORKDIR
RUN chown -R user:user $WORKDIR
RUN chmod 755 $WORKDIR

# RUN nvidia-smi


COPY requirements.txt $WORKDIR/requirements.txt
COPY oneformer $WORKDIR/oneformer
# RUN python3.8 --version
# RUN which python3.8
# RUN python3.8 -m pip install --upgrade pip
# RUN python3.8 -m pip install multidict
# RUN python3.8 -m pip install typing-extensions
# RUN python3.8 -m pip install --upgrade setuptools
# RUN python3.8 -m pip install wheel
# RUN python3.8 -m pip install cython
# RUN python3.8 -m pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

RUN pwd
RUN ls

RUN sh deform_setup.sh

USER user

EXPOSE 7860

ENTRYPOINT ["python", "gradio_app.py"]
