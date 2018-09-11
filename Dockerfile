FROM python:2.7

MAINTAINER pkoperek@gmail.com

RUN mkdir -p /mgr
RUN pip install --no-cache-dir gym torch torchvision ipython
RUN pip install --no-cache-dir -e git+https://github.com/pkoperek/gym_cloudsimplus#egg=gym_cloudsimplus

RUN apt-get update && apt-get install -y netcat

COPY test* /mgr/
COPY wait-for /mgr/

WORKDIR /mgr
CMD ["sleep", "5", "&&", "python", "test_dcnull.py"]
