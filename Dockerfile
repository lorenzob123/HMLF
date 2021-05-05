FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN apt-get update && apt-get install git -y

RUN git clone https://github.com/lorenzob123/HMLF
RUN cd HMLF && pip install -e .[extra] && pip install pytest
