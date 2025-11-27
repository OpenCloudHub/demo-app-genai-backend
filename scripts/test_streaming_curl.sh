#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

API_URL="${1:-http://localhost:8000}"
QUESTION="${2:-What is GitOps in OpenCloudHub?}"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Testing Streaming Response${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Question:${NC} $QUESTION"
echo -e "${GREEN}API URL:${NC} $API_URL"
echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}\n"

START_TIME=$(date +%s)

echo -e "${YELLOW}Response:${NC}\n"

# Parse SSE stream and extract just the content
curl -N -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d "{\"question\": \"$QUESTION\", \"stream\": true}" \
  2>/dev/null \
  | while IFS= read -r line; do
      # Skip empty lines
      if [ -z "$line" ]; then
        continue
      fi

      # Check if line starts with "data: "
      if [[ "$line" == data:* ]]; then
        # Extract content after "data: "
        content="${line#data: }"

        # Check for [DONE] marker
        if [[ "$content" == "[DONE]" ]]; then
          echo -e "\n${CYAN}[Stream completed]${NC}"
        else
          # Print content without newline (streaming effect)
          echo -n "$content"
        fi
      fi
    done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n${BLUE}────────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}Duration:${NC} ${DURATION}s"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
