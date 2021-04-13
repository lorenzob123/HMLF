FROM nvcr.io/nvidia/pytorch:21.03-py3

ARG GIT_ACCESS_TOKEN
RUN apt-get update && apt-get install git -y

RUN git clone https://${GIT_ACCESS_TOKEN}@github.tik.uni-stuttgart.de/IFF/HMLF.git

RUN cd HMLF && pip install -e .