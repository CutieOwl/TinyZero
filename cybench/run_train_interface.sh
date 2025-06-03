#!/bin/bash

# docker-sync start

# Infinite loop
while true; do
    # Run your original script with all passed arguments
    ./run_task.sh "$@"
done

# docker-sync stop
# docker-sync clean