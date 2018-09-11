release:
	docker build -t pkoperek/dqn-manager:latest -t pkoperek/dqn-manager:0.1 .
	docker login
	docker push pkoperek/dqn-manager:latest
