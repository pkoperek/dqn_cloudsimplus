#!/bin/bash

START_DATE=`date +%Y-%m-%dT%H:%M:%S`
echo "Starting simulation at ${START_DATE}"

wait-for-it -s database:5432 -- python3 infinity/deamon.py
