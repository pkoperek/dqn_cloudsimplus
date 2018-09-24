VERSION=0.6

release:
	docker build -t pkoperek/dqn-manager:latest -t pkoperek/dqn-manager:${VERSION} .
	docker push pkoperek/dqn-manager:latest

build:
	docker build -t pkoperek/dqn-manager:latest -t pkoperek/dqn-manager:${VERSION} .
