
# get base image, testing on desktop use ubuntu, on jetson use the L4T lines
#FROM ubuntu:focal

ARG L4T_IMG_TAG=r36.4.0
#FROM nvcr.io/nvidia/l4t-ml:${L4T_IMG_TAG}-py3

# if the above link fails use this instead
FROM dustynv/l4t-ml:${L4T_IMG_TAG}

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app
ENTRYPOINT ["/app/runner.sh"]