
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

WORKDIR /workspace

COPY sources.list /etc/apt/sources.list
RUN apt-get update

RUN apt-get install git -y
RUN apt-get install libgl1-mesa-glx
RUN apt-get install libglib2.0-0
RUN pip install opencv-python 
RUN pip install tensorboard

RUN git clone https://github.com/Polariche/DeepNormals.git