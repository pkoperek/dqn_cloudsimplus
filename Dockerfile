FROM python:2.7

MAINTAINER pkoperek@gmail.com

RUN mkdir -p /mgr
RUN pip install gym torch torchvision
RUN pip install -e git+https://github.com/pkoperek/gym_cloudsimplus#egg=gym_cloudsimplus

COPY test* /mgr/

CMD ["bash"]
