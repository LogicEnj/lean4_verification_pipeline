#!/bin/bash
echo "Waiting for server to start..."
# 2. Get the process ID (PID) of the server
SERVER_PID=$1
LOG_FILENAME=$2

# 3. Wait until the expected log line appears
timeout=100  # Max wait time in seconds
start_time=$(date +%s)

while ! grep -q "INFO:     Uvicorn running on http://0.0.0.0:8000" $LOG_FILENAME; do
    sleep 1  # Check every second

    # Check if the server process crashed
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Error: Server process died!"
        exit 1
    fi

    # Timeout after $timeout seconds
    if [ $(($(date +%s) - start_time)) -ge $timeout ]; then
        echo "Error: Server did not start within $timeout seconds"
        kill -9 $SERVER_PID
        exit 1
    fi
done

# 4. Server is ready!
echo "Server is up and running at http://0.0.0.0:8000"