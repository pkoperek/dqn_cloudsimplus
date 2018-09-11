release:
	docker build -t pkoperek/dqn-manager:latest -t pkoperek/dqn-manager:0.2 .
	docker push pkoperek/dqn-manager:latest
