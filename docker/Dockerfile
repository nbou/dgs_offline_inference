
# get base image, testing on desktop use ubuntu, on jetson use the L4T lines
#FROM ubuntu:focal

ARG L4T_IMG_TAG=r36.4.0
#FROM nvcr.io/nvidia/l4t-ml:${L4T_IMG_TAG}-py3

# if the above link fails use this instead
FROM dustynv/l4t-ml:${L4T_IMG_TAG}

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 -r requirements.txt \
    && pip3 install -e .

# Set the DISPLAY environment variable
ENV DISPLAY=$DISPLAY

#RUN chmod +x /app/runner.sh
#ENTRYPOINT ["/app/runner.sh"]