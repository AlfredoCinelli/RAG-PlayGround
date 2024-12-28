#!bin/bash

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ] && [ ! -f "docker-compose.yaml" ]; then
    echo "Error: docker-compose.yml not found!"
    exit 1
fi

# Start containers using docker compose
echo "Starting containers with docker compose..."
docker compose up -d

# Wait for containers to start
echo "Waiting for containers to start..."
sleep 10

# Get hybrid search config
HYBRID_SEARCH_CONFIG=$(jq '.' conf/hybrid_search_config.json)

echo "Setting Opensearch configs..."
# Set hybridd search config in Opensearch
if curl -XPUT "http://localhost:9200/_search/pipeline/nlp-search-pipeline" -H 'Content-Type: application/json' -d "$HYBRID_SEARCH_CONFIG" > /dev/null 2>&1; then
    sleep 1
    echo "Opensearch is properly configured."
else
    echo "Opensearh is not properly configured."
fi