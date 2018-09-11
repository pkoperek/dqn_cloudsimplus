#!/bin/bash

echo "Waiting for gateway to setup"
sleep 10
START_DATE=`date +%Y-%m-%dT%H:%M:%S`
TEST_FILE=test_${TEST_CASE:-model}.py
echo "Starting simulation: ${TEST_FILE} at ${START_DATE}"
python ${TEST_FILE}
