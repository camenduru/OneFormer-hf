FROM python:3.8.15

RUN useradd -ms /bin/bash admin

ENV WORKDIR=/code
WORKDIR $WORKDIR
RUN chown -R admin:admin $WORKDIR
RUN chmod 755 $WORKDIR

COPY requirements.txt $WORKDIR/requirements.txt

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

RUN sh deform_setup.sh

USER admin

EXPOSE 7860

ENTRYPOINT ["python", "gradio_app.py"]
