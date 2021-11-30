build:
	docker build -t ml-api .

run:
	docker run -v ~/.aws/credentials:/root/.aws/credentials  --env-file ./.env -it -p 80:80  ml-api