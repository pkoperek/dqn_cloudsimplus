FROM python:2.7

MAINTAINER pkoperek@gmail.com

RUN mkdir -p /mgr
RUN pip install --no-cache-dir gym torch torchvision ipython
RUN pip install --no-cache-dir -e git+https://github.com/pkoperek/gym_cloudsimplus#egg=gym_cloudsimplus

RUN apt-get update \
    && apt-get install -y netcat \
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

COPY test* /mgr/
COPY entrypoint.sh /

WORKDIR /mgr
ENTRYPOINT ["/entrypoint.sh"]
