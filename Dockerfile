FROM continuumio/anaconda3:2021.11

ADD . /code 
WORKDIR /code 

# start web framework
ENTRYPOINT ["python", "hockey_app.py"]