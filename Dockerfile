# Dockerfile for Yolov4 pytorch

# Initialize build stage and set up base image
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# Pass User ID and Group ID as arguments. User name can be set to any arbitrary value for this image. 
ARG USER_ID
ARG GROUP_ID
ARG UNAME=tempuser
ARG DEBIAN_FRONTEND=noninteractive

# Setup proxies in enviromment
ENV HTTP_PROXY="http://deessrvproxy.intra.ifm:10176"
ENV HTTPS_PROXY="http://deessrvproxy.intra.ifm:10176"
ENV NO_PROXY="localhost,127.0.0.1,intra.ifm,192.168.0.69"

ENV http_proxy="http://deessrvproxy.intra.ifm:10176"
ENV https_proxy="http://deessrvproxy.intra.ifm:10176"
ENV no_proxy="localhost,127.0.0.1,intra.ifm,192.168.0.69"

# Update and install the required Ubuntu libraries. Note: Base image will be upadted with libraries given below.
RUN apt-get update
RUN apt-get install -y openssh-server build-essential cmake openexr libopenexr-dev libilmbase-dev git openssh-server zlib1g-dev lsb-release htop screen libsm6 libxext6 libxrender-dev nano tmux pkg-config libfreetype6-dev libgl1-mesa-dev 

# Set User ID and Group ID of temporary user. This will make container User ID and Group ID same as host.
RUN groupadd -g ${GROUP_ID} ${UNAME}
RUN useradd -l -u ${USER_ID} -g ${UNAME} ${UNAME}

# This will overwrite base image libraries. 
RUN python -m pip install --upgrade pip
RUN pip install numpy torchvision tensorboard flowpy scikit_image==0.16.2 matplotlib tqdm==4.43.0 easydict==1.9 Pillow==7.1.2 scikit-image opencv-python pycocotools onnxruntime

# Set the user for subsequent commands
USER ${UNAME}
