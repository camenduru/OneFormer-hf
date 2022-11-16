FROM python:3.8.15
FROM nvidia/cuda:11.4-cudnn8-runtime-ubuntu18.04
CMD nvidia-smi

RUN useradd -ms /bin/bash admin

ENV WORKDIR=/code
WORKDIR $WORKDIR
RUN chown -R admin:admin $WORKDIR
RUN chmod 755 $WORKDIR

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt $WORKDIR/requirements.txt

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

RUN pwd
RUN ls

COPY oneformer $WORKDIR/oneformer

RUN sh deform_setup.sh

USER admin

EXPOSE 7860

ENTRYPOINT ["python", "gradio_app.py"]
