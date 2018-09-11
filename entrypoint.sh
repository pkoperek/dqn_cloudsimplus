#!/bin/bash

echo "Waiting for gateway to setup"
sleep 10
START_DATE=`date +%Y-%m-%dT%H:%M:%S`
echo "Starting simulation: ${START_DATE}"
python test_model.py
