#!/bin/bash

docker compose down
docker network prune -f
docker compose up -d
