#!/usr/bin/env bash
# demo/run_demo.sh - ingest demo document and run a chat query
set -e
BASE_URL="http://localhost:8000"
DEMO_FILE="demo/demo_doc.txt"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Please set OPENAI_API_KEY"
  exit 1
fi

echo "Ingesting demo document..."
curl -s -X POST "$BASE_URL/ingest" -F "file=@$DEMO_FILE" -F "source=demo_doc" | jq

echo
sleep 1

echo "Querying agent..."
curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"query":"What is AssistAI?"}' | jq
