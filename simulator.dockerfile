FROM cityflowproject/cityflow:latest

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
     libglib2.0-0 libxext6 libsm6 libxrender1 \
     git mercurial subversion

RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc && \
    pip install msgpack && \
    pip install tensorflow==1.11 && \
    pip install keras==2.1 && \
    pip install networkx && \
    pip install pandas && \
    pip install matplotlib

