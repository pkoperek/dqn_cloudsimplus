## Running

### Environment variables

* `TEST_CASE` - denotes the test case which should be ran as experiment.
  Defaults to `model`. Available values: `model`, `dcnull`.

### docker-compose

Running a simple test with 
* `docker-compose run manager`

### kubernetes

* `kubectl create -f dqn.yml` - create/deploy
* `kubectl logs <POD>` - view logs of the pod
* `kubectl delete -f dqn.yml` - delete the deployment
