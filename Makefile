docker-build:
	docker build --tag pytrt:latest .

docker-run:
	docker run --gpus all --rm -it pytrt