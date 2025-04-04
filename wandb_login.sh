#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Check if the API key is provided
if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY environment variable is not set."
  exit 1
fi

# Run wandb login using the API key
echo "$WANDB_API_KEY" | wandb login
echo "Successfully logged into wandb."
