# kueski-challenge-api
This repo contains the the APIs that will retrieve users' features and infer their status.

### Steps
1. Copy the `sample.env` to a `.env` file
2. Create a bucket named `kueski-ml-system` in S3
3. Configure aws-cli in your host machine (your user must have PUT and READ permissions in S3)
4. Run `make build`
5. Run `make run`