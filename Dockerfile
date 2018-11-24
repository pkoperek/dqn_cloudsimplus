FROM python:3.6

MAINTAINER pkoperek@gmail.com

RUN mkdir -p /mgr
RUN pip3 install --no-cache-dir gym torch torchvision ipython psycopg2-binary
RUN pip3 install --no-cache-dir -e git+https://github.com/pkoperek/gym_cloudsimplus#egg=gym_cloudsimplus

RUN apt-get update \
    && apt-get install -y wait-for-it \
    && apt-get purge --auto-remove -yqq \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

COPY infinity /mgr/infinity
COPY entrypoint.sh /

WORKDIR /mgr
CMD ["/entrypoint.sh"]
