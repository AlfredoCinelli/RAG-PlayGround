#!/bin/bash

# Define Docker container names
docker_container_names=("opensearch" "opensearch-dashboards")

# Check if docker is up
if ! pgrep docker > /dev/null; then
    echo "Docker is not running. Starting Docker..."
    open -a Docker # open Docker Desktop
    sleep 6 # wait for Docker to start
    for container_name in "${docker_container_names[@]}"; do
        echo "Starting specific Docker container $container_name..."
        if docker start "$container_name" >/dev/null 2>&1; then
            echo "Container $container_name started successfully"
        else
            echo "Failed to start container $container_name. Ceating from scratch..."
            # Create Opensearch
            source bash/create_opensearch.sh
        fi
    done
else
    echo "Docker is already running."
    for container_name in "${docker_container_names[@]}"; do
        echo "Starting specific Docker container $container_name..."
        if docker start "$container_name" >/dev/null 2>&1; then
            echo "Container $container_name started successfully"
        else
            echo "Failed to start container $container_name. Creating from scratch..."
            # Create Opensearch
            source bash/create_opensearch.sh
        fi
    done
fi